"""Flower ServerApp for Federated Learning with Differential Privacy."""

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from wisdm_har_dp.model import ActivityCNN
from wisdm_har_dp.task import (
    ActivityDataset,
    initialize_data,
    get_weights,
    set_weights,
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy metrics from all clients using weighted average."""
    if not metrics:
        return {}

    accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0}


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics including privacy budget."""
    if not metrics:
        return {}

    total_examples = sum([num_examples for num_examples, _ in metrics])

    # Weighted average of training loss
    train_losses = [num_examples * m.get("train_loss", 0) for num_examples, m in metrics]
    avg_loss = sum(train_losses) / total_examples if total_examples > 0 else 0

    # Report maximum epsilon across clients (most conservative privacy guarantee)
    epsilons = [m.get("epsilon", 0) for _, m in metrics]
    max_epsilon = max(epsilons) if epsilons else 0

    return {
        "train_loss": avg_loss,
        "max_epsilon": max_epsilon,
    }


class DPFedAvgStrategy(FedAvg):
    """FedAvg strategy with Differential Privacy tracking.

    Extends standard FedAvg:
    - Track cumulative privacy budget across rounds
    - Report privacy metrics after each round
    - Evaluate on centralized test set
    - Save model and history at the end
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        enable_dp: bool,
        num_features: int,
        num_classes: int,
        hidden_size: int,
        test_loader: DataLoader,
        output_dir: Path,
        num_rounds: int,
        **kwargs
    ):
        super().__init__(
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            **kwargs
        )
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.enable_dp = enable_dp
        self.cumulative_epsilon = 0.0
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.num_rounds = num_rounds
        self.device = torch.device("cpu")

        # Create global model for evaluation
        self.global_model = ActivityCNN(
            num_features=num_features,
            num_classes=num_classes,
            hidden_size=hidden_size
        ).to(self.device)

        # Training history
        self.history = {
            'round': [],
            'train_loss': [],
            'test_loss': [],
            'test_acc': [],
            'epsilon': []
        }

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and track privacy budget."""

        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None:
            parameters, metrics = aggregated

            # Update global model
            weights = parameters_to_ndarrays(parameters)
            set_weights(self.global_model, weights)

            # Evaluate on centralized test set
            test_loss, test_acc = self._evaluate_centralized()

            # Track privacy budget
            train_loss = metrics.get("train_loss", 0) if metrics else 0
            epsilon = metrics.get("max_epsilon", 0) if metrics else 0
            self.cumulative_epsilon = epsilon

            # Record history
            self.history['round'].append(server_round)
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['epsilon'].append(epsilon)

            # Print progress
            print(f"\n[Round {server_round}/{self.num_rounds}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss:  {test_loss:.4f}")
            print(f"  Test Acc:   {test_acc:.4f}")
            if self.enable_dp:
                print(f"  Epsilon:    {epsilon:.2f}")
                if epsilon > self.target_epsilon:
                    print(f" Privacy budget exceeded target (ε={self.target_epsilon})")

            # Save on last round
            if server_round == self.num_rounds:
                self._save_results()

        return aggregated

    def _evaluate_centralized(self) -> Tuple[float, float]:
        """Evaluate global model on centralized test set."""
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * len(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _save_results(self):
        """Save model and training history."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_params = {name: param.cpu().numpy()
                        for name, param in self.global_model.state_dict().items()}
        model_path = self.output_dir / 'federated_dp_model.npy'
        np.save(model_path, model_params)
        print(f"\n✓ Model saved to {model_path}")

        # Save history
        history_path = self.output_dir / 'federated_training_history.npy'
        np.save(history_path, self.history)
        print(f"✓ Training history saved to {history_path}")

        # Print final summary
        print("Training Complete!")
        print(f"Final Test Accuracy: {self.history['test_acc'][-1]:.4f}")
        if self.enable_dp:
            print(f"Final Privacy Budget: (ε={self.cumulative_epsilon:.2f}, δ={self.target_delta})")
            print(f"\nPrivacy Interpretation:")
            print(f"  An attacker cannot distinguish whether a specific sample")
            print(f"  was in the training set with probability greater than")
            print(f"  e^ε ≈ {np.exp(self.cumulative_epsilon):.2f} times random guessing.")


def server_fn(context) -> ServerAppComponents:
    """Create ServerAppComponents for the Flower server."""

    run_config = context.run_config

    # Get configuration
    num_rounds = int(run_config.get("num-server-rounds", 10))
    fraction_fit = float(run_config.get("fraction-fit", 1.0))
    fraction_evaluate = float(run_config.get("fraction-evaluate", 1.0))
    batch_size = int(run_config.get("batch-size", 32))
    hidden_size = int(run_config.get("hidden-size", 64))
    enable_dp = bool(run_config.get("enable-dp", True))
    target_epsilon = float(run_config.get("target-epsilon", 8.0))
    target_delta = float(run_config.get("target-delta", 1e-5))
    dataset_path = run_config.get("dataset-path", "WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.csv")
    num_supernodes = int(run_config.get("num-supernodes", 3))

    # Initialize data
    data_state = initialize_data(dataset_path, num_supernodes)

    # Create centralized test loader
    test_dataset = ActivityDataset(data_state["X_test"], data_state["y_test"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create initial model and get parameters
    initial_model = ActivityCNN(
        num_features=data_state["num_features"],
        num_classes=data_state["num_classes"],
        hidden_size=hidden_size
    )
    initial_parameters = ndarrays_to_parameters(get_weights(initial_model))

    # Output directory
    output_dir = Path("outputs")

    # Create strategy
    strategy = DPFedAvgStrategy(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        enable_dp=enable_dp,
        num_features=data_state["num_features"],
        num_classes=data_state["num_classes"],
        hidden_size=hidden_size,
        test_loader=test_loader,
        output_dir=output_dir,
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_supernodes,
        min_evaluate_clients=num_supernodes,
        min_available_clients=num_supernodes,
        initial_parameters=initial_parameters,
    )

    # Print configuration
    print("Federated Learning with Differential Privacy:")
    print(f"  Clients: {num_supernodes}")
    print(f"  Rounds:  {num_rounds}")
    print(f"  DP:      {'Enabled' if enable_dp else 'Disabled'}")
    if enable_dp:
        print(f"  Target ε: {target_epsilon}")
        print(f"  Target δ: {target_delta}")

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create the Flower ServerApp
app = ServerApp(server_fn=server_fn)
