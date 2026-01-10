"""WISDM-HAR-Classification: A Flower / pytorch_legacy_api app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.types import Number

from wisdm_har_classification.dataloader import load_data
from wisdm_har_classification.logger import logger
from wisdm_har_classification.model import Net, get_weights, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        metrics = train(self.net, self.trainloader, self.local_epochs, self.device)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)

        metrics = test(self.net, self.valloader, self.device)

        return (
            metrics["loss"],
            len(self.valloader.dataset),
            metrics,
        )


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    local_epochs = int(context.run_config["local-epochs"])

    net = Net()
    trainloader, valloader = load_data(partition_id, num_partitions)

    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
