"""Post-training visualization script.

Run after `flwr run .` completes to generate plots.
python create_plots.py [--history-path PATH] [--output-dir PATH]
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from wisdm_har_dp.model import ActivityCNN
from wisdm_har_dp.task import ActivityDataset, initialize_data, set_weights


def load_history(history_path: Path) -> dict:
    """Load training history from .npy file."""
    history = np.load(history_path, allow_pickle=True).item()
    print(f"Loaded history with {len(history['round'])} rounds")
    return history


def load_model(model_path: Path, num_features: int, num_classes: int, hidden_size: int = 64):
    """Load model from .npy file."""
    model_params = np.load(model_path, allow_pickle=True).item()
    model = ActivityCNN(num_features, num_classes, hidden_size)

    # Convert numpy arrays to tensors and load
    state_dict = {k: torch.tensor(v) for k, v in model_params.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def create_visualizations(history: dict, predictions: np.ndarray, targets: np.ndarray,
                          class_names: list, target_epsilon: float, enable_dp: bool,
                          output_dir: Path):
    """Create and save training visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training curves - Loss
    ax1 = axes[0, 0]
    ax1.plot(history['round'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(history['round'], history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(history['round'], history['test_acc'], 'g-', label='Test Acc', linewidth=2)
    ax2.set_xlabel('Federated Round')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy over Federated Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Privacy budget
    ax3 = axes[1, 0]
    if enable_dp and 'epsilon' in history:
        ax3.plot(history['round'], history['epsilon'], 'purple', linewidth=2)
        ax3.axhline(y=target_epsilon, color='r', linestyle='--',
                   label=f'Target ε={target_epsilon}')
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
                xticklabels=class_names,
                yticklabels=class_names, ax=ax4)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix')

    plt.tight_layout()
    fig_path = output_dir / 'federated_training_results.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to {fig_path}")

    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Create visualizations from FL training results')
    parser.add_argument('--history-path', type=str, default='outputs/federated_training_history.npy',
                        help='Path to training history .npy file')
    parser.add_argument('--model-path', type=str, default='outputs/federated_dp_model.npy',
                        help='Path to model .npy file')
    parser.add_argument('--dataset-path', type=str, default='WISDM_ar_v1.1/client_B.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for plots')
    parser.add_argument('--target-epsilon', type=float, default=8.0,
                        help='Target epsilon for DP')
    parser.add_argument('--enable-dp', type=bool, default=True,
                        help='Whether DP was enabled')
    parser.add_argument('--num-clients', type=int, default=3,
                        help='Number of clients used in training')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load history
    history_path = Path(args.history_path)
    if not history_path.exists():
        print(f"ERROR: History file not found: {history_path}")
        print("Make sure training completed and saved the history file.")
        return 1

    history = load_history(history_path)

    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Rounds completed: {len(history['round'])}")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    if args.enable_dp and 'epsilon' in history:
        print(f"Final Epsilon: {history['epsilon'][-1]:.2f}")

    # Load data to get class names and test set
    print("\nLoading data for evaluation...")
    data_state = initialize_data(args.dataset_path, args.num_clients)
    class_names = list(data_state['label_encoder'].classes_)

    # Load model and get predictions
    model_path = Path(args.model_path)
    if model_path.exists():
        model = load_model(
            model_path,
            num_features=data_state['num_features'],
            num_classes=data_state['num_classes']
        )

        # Create test loader
        test_dataset = ActivityDataset(data_state['X_test'], data_state['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Get predictions
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.numpy())
                all_targets.extend(target.numpy())

        predictions = np.array(all_preds)
        targets = np.array(all_targets)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(targets, predictions, target_names=class_names))
    else:
        print(f"WARNING: Model file not found: {model_path}")
        print("Using dummy predictions for visualization...")
        # Use random predictions as placeholder
        predictions = np.zeros(100, dtype=int)
        targets = np.zeros(100, dtype=int)

    # Create visualizations
    create_visualizations(
        history=history,
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        target_epsilon=args.target_epsilon,
        enable_dp=args.enable_dp,
        output_dir=output_dir
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
