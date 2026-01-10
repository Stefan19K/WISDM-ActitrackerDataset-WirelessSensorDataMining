import os
from flwr.server.typing import ServerAppCallable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from wisdm_har_classification.logger import logger

PLOTS = "plots"
METRICS = "metrics"
SAVED_CONFIGS = "saved_configs"
CLASSES = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

def save_plots(df, save_dir):
    # Ensure dir exists
    os.makedirs(save_dir, exist_ok=True)

    # 1. Plot Loss
    plt.figure(figsize=(10, 6))
    if 't_loss' in df.columns:
        plt.plot(df['round'], df['t_loss'], label='Train Loss')
    if 'v_loss' in df.columns:
        plt.plot(df['round'], df['v_loss'], label='Validation Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    # 2. Plot Accuracy
    plt.figure(figsize=(10, 6))
    if 't_acc' in df.columns:
        plt.plot(df['round'], df['t_acc'], label='Train Accuracy')
    if 'v_acc' in df.columns:
        plt.plot(df['round'], df['v_acc'], label='Validation Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f"{save_dir}/accuracy_curve.png")
    plt.close()

    # 3. Confusion Matrix (Last Round)
    cm_cols = [c for c in df.columns if c.startswith('v_cm_')]
    if cm_cols:
        # Sort columns to ensure correct order 0_0, 0_1, ...
        # The aggregation might mess up order if dict iteration was random, but usually it's stable-ish or we should sort.
        # cm_i_j. Let's sort by i then j.
        # c.split('_') -> ['validation', 'cm', '0', '0']
        cm_cols.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3])))

        last_round_idx = df.index[-1]
        cm_values = df.loc[last_round_idx, cm_cols].values.astype(float)

        num_classes = int(np.sqrt(len(cm_values)))
        if num_classes * num_classes == len(cm_values):
            cm = cm_values.reshape(num_classes, num_classes)

            plt.figure(figsize=(10, 8))
            # Check if CLASSES matches num_classes
            labels = CLASSES if len(CLASSES) == num_classes else [str(i) for i in range(num_classes)]

            sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                        xticklabels=labels,
                        yticklabels=labels)
            plt.xlabel('Predicted Action')
            plt.ylabel('True Action')
            plt.title('Confusion Matrix (Count) - Last Round')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/confusion_matrix.png")
            plt.close()
    # 4. Metrics Table Image
    cols_to_show = [c for c in df.columns if not c.startswith('v_cm_') and not c.startswith('t_cm_')]
    table_df = df[cols_to_show].copy()

    # Create plot for table
    fig, ax = plt.subplots(figsize=(40, len(table_df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')

    table_data = table_df.round(4).astype(str).values
    col_labels = table_df.columns

    the_table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.8)

    # Highlight best validation accuracy
    if 'v_acc' in cols_to_show:
        acc_idx = cols_to_show.index('v_acc')
        best_row_idx = table_df['v_acc'].astype(float).idxmax()

        for (row, col), cell in the_table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#dddddd')
            else:
                df_row_idx = row - 1
                if df_row_idx == best_row_idx:
                     cell.set_facecolor('#ffff99')
                     # Bold the accuracy value
                     if col == acc_idx:
                         cell.set_text_props(weight='bold', color='green')

    plt.title("Federated Metrics History (Highlighted Best Validation Accuracy Round)")

    # Add Legend
    legend_text = (
        "Legend:\n"
        "t: Train, v: Validation\n"
        "acc: Accuracy, loss: Loss\n"
        "p: Precision, r: Recall, f1: F1 Score"
    )
    plt.figtext(0.5, 0.05, legend_text, ha="center", fontsize=12,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    plt.savefig(f"{save_dir}/metrics_table.jpg", bbox_inches='tight', dpi=150)
    plt.close()

class LoggingFedAvg(FedAvg):
    def __init__(self, *args, num_rounds=10, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_rounds = num_rounds

        # Store metrics
        self.best_params = None
        self.best_loss = None
        self.best_round = None
        self.history_fit = {}
        self.history_eval = {}

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None, {}

        aggregated_params, metrics = aggregated
        if metrics is not None:
            for key, val in metrics.items():
                self.history_fit.setdefault(key, []).append((server_round, val))

                if key == 'loss':
                    if self.best_loss is None or val < self.best_loss:
                        self.best_loss = val
                        self.best_params = aggregated_params
                        self.best_round = server_round

        # Save final model at the last round
        if server_round == self.num_rounds:
            # Already imported at top level
            nds = parameters_to_ndarrays(self.best_params)

            if not os.path.exists(SAVED_CONFIGS):
                os.makedirs(SAVED_CONFIGS)
                print(f"Created directory: {SAVED_CONFIGS}")

            np.save(f"{SAVED_CONFIGS}/final_global_model.npy", np.array(nds, dtype=object))

            logger.info(f"Saved final global model at round {server_round}")

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            return None, {}

        loss, metrics = aggregated
        if metrics is not None:
            for key, val in metrics.items():
                self.history_eval.setdefault(key, []).append((server_round, val))

        if server_round == self.num_rounds:
            df = self.to_dataframe()

            if not os.path.exists(METRICS):
                os.makedirs(METRICS)
                print(f"Created directory: {METRICS}")

            df.to_csv(f"{METRICS}/flwr_metrics.csv", index=False)
            print("\nSaved federated metrics to flwr_metrics.csv\n")
            print(df)

            # Generate plots
            try:
                if not os.path.exists(PLOTS):
                    os.makedirs(PLOTS)
                    print(f"Created directory: {PLOTS}")

                save_plots(df, PLOTS)
                print(f"\nSaved plots to {PLOTS}/\n")
            except Exception as e:
                print(f"\nError saving plots: {e}\n")

        return aggregated

    def to_dataframe(self):
        rows = {}

        # Training metrics
        for metric, items in self.history_fit.items():
            for rnd, val in items:
                rows.setdefault(rnd, {})[f"t_{metric}"] = val

        # Validation metrics
        for metric, items in self.history_eval.items():
            for rnd, val in items:
                rows.setdefault(rnd, {})[f"v_{metric}"] = val

        df = (
            pd.DataFrame.from_dict(rows, orient="index")
            .sort_index()
            .reset_index()
            .rename(columns={"index": "round"})
        )

        return df
