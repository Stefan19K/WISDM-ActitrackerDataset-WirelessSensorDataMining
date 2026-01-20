"""Flower ClientApp for Federated Learning with Differential Privacy."""

import torch
from torch.utils.data import DataLoader

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from wisdm_har_dp.model import ActivityCNN
from wisdm_har_dp.task import (
    ActivityDataset,
    get_client_data,
    get_weights,
    set_weights,
    train,
    evaluate,
    initialize_data,
)


class FlowerClient(NumPyClient):
    """Flower client with Differential Privacy support.

    Each client:
    - Maintains local model and data
    - Trains with DP-SGD (clipped gradients + noise)
    - Reports privacy budget consumption
    """

    def __init__(self, partition_id: int, config: dict):
        self.partition_id = partition_id
        self.config = config
        self.device = torch.device("cpu")  # Force CPU for Ray compatibility

        # Get data for this partition
        train_data, test_data = get_client_data(partition_id)

        # Create datasets and loaders
        self.train_dataset = ActivityDataset(train_data[0], train_data[1])
        self.test_dataset = ActivityDataset(test_data[0], test_data[1])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config["batch_size"]
        )

        # Initialize model
        num_features = train_data[0].shape[1]
        self.model = ActivityCNN(
            num_features=num_features,
            num_classes=config["num_classes"],
            hidden_size=config["hidden_size"]
        ).to(self.device)

        # Track total steps for epsilon computation
        self.total_steps = 0

        print(f"Client {partition_id} initialized with {len(self.train_dataset)} training samples")

    def get_parameters(self, config):
        """Return model parameters as numpy arrays."""
        return get_weights(self.model)

    def fit(self, parameters, config):
        """Train model with Differential Privacy."""
        set_weights(self.model, parameters)

        loss, num_samples, epsilon, self.total_steps = train(
            model=self.model,
            train_loader=self.train_loader,
            config=self.config,
            device=self.device,
            total_steps=self.total_steps
        )

        metrics = {
            "train_loss": float(loss),
            "epsilon": float(epsilon),
            "partition_id": self.partition_id,
        }

        if self.config["enable_dp"]:
            print(f"Client {self.partition_id} - Loss: {loss:.4f}, ε: {epsilon:.2f}")
        else:
            print(f"Client {self.partition_id} - Loss: {loss:.4f}")

        return get_weights(self.model), num_samples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        set_weights(self.model, parameters)

        loss, accuracy, num_samples = evaluate(
            model=self.model,
            test_loader=self.test_loader,
            device=self.device
        )

        return float(loss), num_samples, {"accuracy": float(accuracy)}


def client_fn(context: Context):
    """Create a Flower client for the given partition."""
    # Get partition ID from context
    partition_id = context.node_config["partition-id"]

    # Get config from run_config
    run_config = context.run_config

    # Initialize data if not already done
    dataset_path = run_config.get("dataset-path", "WISDM_ar_v1.1/client_B.csv")
    num_supernodes = int(run_config.get("num-supernodes", 3))

    data_state = initialize_data(dataset_path, num_supernodes)

    # Build config dict for client
    config = {
        "batch_size": int(run_config.get("batch-size", 32)),
        "local_epochs": int(run_config.get("local-epochs", 3)),
        "learning_rate": float(run_config.get("learning-rate", 0.001)),
        "hidden_size": int(run_config.get("hidden-size", 64)),
        "enable_dp": bool(run_config.get("enable-dp", True)),
        "target_epsilon": float(run_config.get("target-epsilon", 8.0)),
        "target_delta": float(run_config.get("target-delta", 1e-5)),
        "max_grad_norm": float(run_config.get("max-grad-norm", 1.0)),
        "noise_multiplier": float(run_config.get("noise-multiplier", 0.5)),
        "num_classes": data_state["num_classes"],
        "num_features": data_state["num_features"],
    }

    # Return Client (not NumPyClient) to avoid deprecation warning
    return FlowerClient(partition_id=partition_id, config=config).to_client()


# Create the Flower ClientApp
app = ClientApp(client_fn=client_fn)
