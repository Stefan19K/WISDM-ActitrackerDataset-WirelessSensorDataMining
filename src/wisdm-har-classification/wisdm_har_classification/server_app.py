"""Flower ServerApp for Federated Learning with Differential Privacy."""

# Module-level debug - write immediately when module loads
from pathlib import Path
_debug_dir = Path("outputs")
_debug_dir.mkdir(parents=True, exist_ok=True)
with open(_debug_dir / "server_module_loaded.txt", "w") as f:
    f.write("server_app.py module was imported\n")

from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import flwr as fl
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
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
    """
    FedAvg strategy with Differential Privacy tracking.

    Extends standard FedAvg to:
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
        label_encoder,
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
        self.label_encoder = label_encoder
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

        # File-based logging since stdout may be suppressed
        with open(self.output_dir / "training_log.txt", "a") as log:
            log.write(f"\n=== Round {server_round} ===\n")

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

            # Log to file
            with open(self.output_dir / "training_log.txt", "a") as log:
                log.write(f"Train Loss: {train_loss:.4f}\n")
                log.write(f"Test Loss:  {test_loss:.4f}\n")
                log.write(f"Test Acc:   {test_acc:.4f}\n")
                if self.enable_dp:
                    log.write(f"Epsilon:    {epsilon:.2f}\n")

            # Also print (may not show, but try anyway)
            print(f"\n[Round {server_round}/{self.num_rounds}]", flush=True)
            print(f"  Train Loss: {train_loss:.4f}", flush=True)
            print(f"  Test Loss:  {test_loss:.4f}", flush=True)
            print(f"  Test Acc:   {test_acc:.4f}", flush=True)

            # Save history after EVERY round (in case we crash)
            np.save(self.output_dir / 'federated_training_history.npy', self.history)

            # Save on last round
            if server_round == self.num_rounds:
                with open(self.output_dir / "training_log.txt", "a") as log:
                    log.write(f"\n=== FINAL ROUND - Saving results ===\n")
                try:
                    self._save_results()
                except Exception as e:
                    with open(self.output_dir / "training_log.txt", "a") as log:
                        log.write(f"ERROR saving results: {e}\n")
                        import traceback
                        log.write(traceback.format_exc())

        return aggregated

    def _evaluate_centralized(self, return_predictions: bool = False):
        """Evaluate global model on centralized test set."""
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * len(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

                if return_predictions:
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        if return_predictions:
            return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)
        return avg_loss, accuracy

    def _save_results(self):
        """Save model, training history, and visualizations."""
        print(f">>> _save_results called, output_dir: {self.output_dir}", flush=True)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f">>> Output directory created/verified", flush=True)

            # Get final predictions for confusion matrix
            _, _, predictions, targets = self._evaluate_centralized(return_predictions=True)
            print(f">>> Got predictions: {len(predictions)} samples", flush=True)

            # Save model
            model_params = {name: param.cpu().numpy()
                            for name, param in self.global_model.state_dict().items()}
            model_path = self.output_dir / 'federated_dp_model.npy'
            np.save(model_path, model_params)
            print(f"✓ Model saved to {model_path}", flush=True)

            # Save history
            history_path = self.output_dir / 'federated_training_history.npy'
            np.save(history_path, self.history)
            print(f"✓ Training history saved to {history_path}", flush=True)

            # Create visualizations
            self._create_visualizations(predictions, targets)

            # Print classification report
            print(f"\nClassification Report:", flush=True)
            print(classification_report(targets, predictions, target_names=self.label_encoder.classes_), flush=True)

            # Print final summary
            print("\n" + "=" * 60, flush=True)
            print("Training Complete!", flush=True)
            print("=" * 60, flush=True)
            print(f"Final Test Accuracy: {self.history['test_acc'][-1]:.4f}", flush=True)
            if self.enable_dp:
                print(f"Final Privacy Budget: (ε={self.cumulative_epsilon:.2f}, δ={self.target_delta})", flush=True)
                print(f"\nPrivacy Interpretation:", flush=True)
                print(f"  An attacker cannot distinguish whether a specific sample", flush=True)
                print(f"  was in the training set with probability greater than", flush=True)
                print(f"  e^ε ≈ {np.exp(self.cumulative_epsilon):.2f} times random guessing.", flush=True)
        except Exception as e:
            print(f"ERROR in _save_results: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _create_visualizations(self, predictions: np.ndarray, targets: np.ndarray):
        """Create and save training visualizations."""
        try:
            print(f">>> Creating visualizations...", flush=True)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Training curves - Loss
            ax1 = axes[0, 0]
            ax1.plot(self.history['round'], self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax1.plot(self.history['round'], self.history['test_loss'], 'r-', label='Test Loss', linewidth=2)
            ax1.set_xlabel('Federated Round')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Test Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Accuracy curves
            ax2 = axes[0, 1]
            ax2.plot(self.history['round'], self.history['test_acc'], 'g-', label='Test Acc', linewidth=2)
            ax2.set_xlabel('Federated Round')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Test Accuracy over Federated Rounds')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Privacy budget
            ax3 = axes[1, 0]
            if self.enable_dp:
                ax3.plot(self.history['round'], self.history['epsilon'], 'purple', linewidth=2)
                ax3.axhline(y=self.target_epsilon, color='r', linestyle='--',
                           label=f'Target ε={self.target_epsilon}')
                ax3.set_xlabel('Federated Round')
                ax3.set_ylabel('Privacy Budget (ε)')
                ax3.set_title('Cumulative Privacy Budget')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'DP Disabled', ha='center', va='center', fontsize=14)
                ax3.set_title('Privacy Budget (DP Disabled)')

            # Confusion matrix
            ax4 = axes[1, 1]
            cm = confusion_matrix(targets, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_, ax=ax4)
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')
            ax4.set_title('Confusion Matrix')

            plt.tight_layout()
            fig_path = self.output_dir / 'federated_training_results.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Visualization saved to {fig_path}", flush=True)
        except Exception as e:
            print(f"ERROR creating visualization: {e}", flush=True)
            import traceback
            traceback.print_exc()


def server_fn(context) -> ServerAppComponents:
    """Create ServerAppComponents for the Flower server."""

    # Create output directory and log immediately
    from pathlib import Path
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "server_fn_called.txt", "w") as f:
        f.write("server_fn was called\n")

    run_config = context.run_config

    with open(output_dir / "server_fn_called.txt", "a") as f:
        f.write(f"run_config: {dict(run_config)}\n")

    # Get configuration
    num_rounds = int(run_config.get("num-server-rounds", 10))
    fraction_fit = float(run_config.get("fraction-fit", 1.0))
    fraction_evaluate = float(run_config.get("fraction-evaluate", 1.0))
    batch_size = int(run_config.get("batch-size", 32))
    hidden_size = int(run_config.get("hidden-size", 64))
    enable_dp = bool(run_config.get("enable-dp", True))
    target_epsilon = float(run_config.get("target-epsilon", 8.0))
    target_delta = float(run_config.get("target-delta", 1e-5))
    dataset_path = run_config.get("dataset-path", "WISDM_ar_v1.1/client_B.csv")
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
        label_encoder=data_state["label_encoder"],
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_supernodes,
        min_evaluate_clients=num_supernodes,
        min_available_clients=num_supernodes,
        initial_parameters=initial_parameters,
    )

    # Print configuration
    print("\n" + "=" * 60, flush=True)
    print("Federated Learning with Differential Privacy", flush=True)
    print("=" * 60, flush=True)
    print(f"  Clients: {num_supernodes}", flush=True)
    print(f"  Rounds:  {num_rounds}", flush=True)
    print(f"  DP:      {'Enabled' if enable_dp else 'Disabled'}", flush=True)
    if enable_dp:
        print(f"  Target ε: {target_epsilon}", flush=True)
        print(f"  Target δ: {target_delta}", flush=True)
    print("=" * 60 + "\n", flush=True)

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create the Flower ServerApp
app = ServerApp(server_fn=server_fn)