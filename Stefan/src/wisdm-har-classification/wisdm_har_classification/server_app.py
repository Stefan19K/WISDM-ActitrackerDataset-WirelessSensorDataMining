"""WISDM Federated Learning Server"""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from wisdm_har_classification.model import Net, get_weights
from wisdm_har_classification.utils import LoggingFedAvg


def agg_metrics(metrics):
    if not metrics:
        return {}

    # extract all keys from the first client
    metric_keys = list(metrics[0][1].keys())

    # prepare accumulators
    totals = {k: 0.0 for k in metric_keys}
    total_samples = 0

    for num_samples, m in metrics:
        total_samples += num_samples
        for k in metric_keys:
            if k in m:
                if k.startswith("cm_"):
                    # Sum counts for confusion matrix
                    totals[k] += float(m[k])
                else:
                    # Weighted sum for other metrics
                    totals[k] += float(m[k]) * num_samples

    # compute final values
    aggregated = {}
    for k in metric_keys:
        if k.startswith("cm_"):
            aggregated[k] = totals[k]
        else:
            aggregated[k] = totals[k] / total_samples

    return aggregated


global_strategy = None


def server_fn(context: Context):
    global global_strategy

    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    global_strategy = LoggingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=agg_metrics,
        fit_metrics_aggregation_fn=agg_metrics,
        num_rounds=num_rounds,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=global_strategy, config=config)


app = ServerApp(server_fn=server_fn)
