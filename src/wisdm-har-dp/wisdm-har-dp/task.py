"""Data loading and training utilities."""

import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from wisdm_har_dp.model import ActivityCNN
from wisdm_har_dp.dp_utils import DPOptimizer, compute_epsilon


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples.

    Focuses training on hard-to-classify examples by down-weighting easy examples.
    Mainly useful for confused classes like Walking/Upstairs/Downstairs.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, num_classes=6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                           self.label_smoothing / self.num_classes
            ce_loss = -(smooth_targets * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


# Global state for data partitioning (shared across client instances)
_data_state = {
    "df": None,
    "label_encoder": None,
    "num_features": None,
    "num_classes": None,
    "users": None,
    "client_users_map": None,
    "X_test": None,
    "y_test": None,
    "class_weights": None,
}


class ActivityDataset(Dataset):
    """PyTorch Dataset for Human Activity Recognition."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def engineer_features(df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, list]:
    """Add engineered features to help distinguish Walking/Upstairs/Downstairs.

    - Stairs have more vertical (Y-axis) variation
    - Upstairs vs Downstairs differ in Y-axis mean (gravity direction)
    - Walking has more regular periodicity
    """
    df = df.copy()
    new_features = []

    # Identify axis-specific columns
    x_cols = [c for c in feature_cols if '_X' in c.upper() or c.startswith('X')]
    y_cols = [c for c in feature_cols if '_Y' in c.upper() or c.startswith('Y')]
    z_cols = [c for c in feature_cols if '_Z' in c.upper() or c.startswith('Z')]

    # Also try without underscore
    if not x_cols:
        x_cols = [c for c in feature_cols if 'X' in c.upper()]
        y_cols = [c for c in feature_cols if 'Y' in c.upper()]
        z_cols = [c for c in feature_cols if 'Z' in c.upper()]

    print(f"Feature engineering - Found axis columns: X={len(x_cols)}, Y={len(y_cols)}, Z={len(z_cols)}")

    # Find mean/avg columns for each axis
    x_mean = [c for c in x_cols if 'mean' in c.lower() or 'avg' in c.lower()]
    y_mean = [c for c in y_cols if 'mean' in c.lower() or 'avg' in c.lower()]
    z_mean = [c for c in z_cols if 'mean' in c.lower() or 'avg' in c.lower()]

    # Find std/var columns
    x_std = [c for c in x_cols if 'std' in c.lower() or 'var' in c.lower()]
    y_std = [c for c in y_cols if 'std' in c.lower() or 'var' in c.lower()]
    z_std = [c for c in z_cols if 'std' in c.lower() or 'var' in c.lower()]

    # Vertical dominance ratio (Y variance / total variance)
    # Stairs have higher vertical variation
    if y_std and x_std and z_std:
        y_var = df[y_std[0]] if 'std' not in y_std[0].lower() else df[y_std[0]]**2
        x_var = df[x_std[0]] if 'std' not in x_std[0].lower() else df[x_std[0]]**2
        z_var = df[z_std[0]] if 'std' not in z_std[0].lower() else df[z_std[0]]**2

        total_var = x_var + y_var + z_var + 1e-8
        df['vertical_dominance'] = y_var / total_var
        df['horizontal_dominance'] = (x_var + z_var) / total_var
        new_features.extend(['vertical_dominance', 'horizontal_dominance'])

    # Y-axis mean (distinguishes upstairs vs downstairs due to gravity)
    if y_mean:
        # Upstairs: leaning forward (negative Y mean)
        # Downstairs: leaning backward (positive Y mean)
        df['y_mean_sign'] = np.sign(df[y_mean[0]])
        df['y_mean_abs'] = np.abs(df[y_mean[0]])
        new_features.extend(['y_mean_sign', 'y_mean_abs'])

    # Signal magnitude area (overall activity level)
    if x_mean and y_mean and z_mean:
        df['signal_magnitude'] = np.sqrt(
            df[x_mean[0]]**2 + df[y_mean[0]]**2 + df[z_mean[0]]**2
        )
        new_features.append('signal_magnitude')

    # Axis ratios
    if x_std and y_std:
        df['y_x_var_ratio'] = (df[y_std[0]]**2) / (df[x_std[0]]**2 + 1e-8)
        new_features.append('y_x_var_ratio')

    if z_std and y_std:
        df['y_z_var_ratio'] = (df[y_std[0]]**2) / (df[z_std[0]]**2 + 1e-8)
        new_features.append('y_z_var_ratio')

    # Find min/max columns for range features
    x_min = [c for c in x_cols if 'min' in c.lower()]
    x_max = [c for c in x_cols if 'max' in c.lower()]
    y_min = [c for c in y_cols if 'min' in c.lower()]
    y_max = [c for c in y_cols if 'max' in c.lower()]

    # Range features (peak-to-peak)
    if x_min and x_max:
        df['x_range'] = df[x_max[0]] - df[x_min[0]]
        new_features.append('x_range')

    if y_min and y_max:
        df['y_range'] = df[y_max[0]] - df[y_min[0]]
        new_features.append('y_range')

    # Asymmetry features (difference between positive and negative peaks)
    if y_min and y_max:
        df['y_asymmetry'] = df[y_max[0]] + df[y_min[0]]  # Sum indicates bias direction
        new_features.append('y_asymmetry')

    # Coefficient of variation (std/mean) - regularity measure
    if x_mean and x_std:
        df['x_cv'] = df[x_std[0]] / (np.abs(df[x_mean[0]]) + 1e-8)
        new_features.append('x_cv')

    if y_mean and y_std:
        df['y_cv'] = df[y_std[0]] / (np.abs(df[y_mean[0]]) + 1e-8)
        new_features.append('y_cv')

    # Interaction features between key statistics
    # These capture combined patterns
    if x_std and y_std:
        df['xy_std_product'] = df[x_std[0]] * df[y_std[0]]
        new_features.append('xy_std_product')

    if y_mean and y_std:
        df['y_mean_std_ratio'] = df[y_mean[0]] / (df[y_std[0]] + 1e-8)
        new_features.append('y_mean_std_ratio')

    # Update feature columns
    all_feature_cols = feature_cols + new_features
    print(f"Feature engineering - Added {len(new_features)} new features: {new_features}")
    print(f"Feature engineering - Total features: {len(all_feature_cols)}")

    return df, all_feature_cols


def initialize_data(dataset_path: str, num_clients: int):
    """Initialize global data state (called once on server/before simulation).

    Use user-based partitioning like wisdm-har-classification.
    """
    global _data_state

    if _data_state["df"] is None:
        print(f"Loading data from {dataset_path}...")
        df = pd.read_csv(dataset_path)

        # Display dataset info
        print(f"Dataset shape: {df.shape}")
        print(f"Classes: {df['class'].unique()}")

        # Get unique users and sort
        users = df["user"].unique()
        users.sort()
        print(f"Total users: {len(users)}")

        # Split users among clients (user-based partitioning for realistic non-IID)
        client_users_map = np.array_split(users, num_clients)
        for i, client_users in enumerate(client_users_map):
            print(f"Client {i} assigned users: {list(client_users)} ({len(client_users)} users)")

        # Label encoder (fit on all data)
        label_encoder = LabelEncoder()
        label_encoder.fit(df['class'].values)
        print(f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

        # Get original feature columns
        drop_cols = ['class', 'user', 'UNIQUE_ID']
        original_feature_cols = [c for c in df.columns if c not in drop_cols]
        print(f"Original features: {len(original_feature_cols)}")

        # Apply feature engineering
        df, feature_cols = engineer_features(df, original_feature_cols)

        num_features = len(feature_cols)
        num_classes = len(label_encoder.classes_)

        print(f"Total features after engineering: {num_features}")
        print(f"Number of classes: {num_classes}")

        # Create centralized test set (20% of data, stratified)
        X_all = df[feature_cols].values
        y_all = label_encoder.transform(df['class'].values)

        _, X_test, _, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        # Scale test set (will be re-scaled per client, but need for centralized eval)
        scaler = StandardScaler()
        scaler.fit(X_all)
        X_test_scaled = scaler.transform(X_test)

        # Compute class weights for handling imbalance
        class_counts = np.bincount(y_all, minlength=num_classes)
        total_samples = len(y_all)

        # Inverse frequency weighting
        class_weights = total_samples / (num_classes * class_counts + 1e-6)
        # Normalize so max weight = 2.0
        class_weights = class_weights / class_weights.max() * 2.0

        print(f"Class distribution: {dict(zip(label_encoder.classes_, class_counts))}")
        print(f"Class weights: {dict(zip(label_encoder.classes_, class_weights.round(3)))}")

        _data_state["df"] = df
        _data_state["feature_cols"] = feature_cols
        _data_state["label_encoder"] = label_encoder
        _data_state["num_features"] = num_features
        _data_state["num_classes"] = num_classes
        _data_state["users"] = users
        _data_state["client_users_map"] = client_users_map
        _data_state["X_test"] = X_test_scaled
        _data_state["y_test"] = y_test
        _data_state["class_weights"] = class_weights

    return _data_state


def get_client_data(partition_id: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Get training and test data for a specific client partition.

    Uses user-based partitioning and applies SMOTE for class balancing.
    """
    global _data_state

    df = _data_state["df"]
    feature_cols = _data_state["feature_cols"]
    label_encoder = _data_state["label_encoder"]
    client_users_map = _data_state["client_users_map"]

    # Get users assigned to this client
    my_users = client_users_map[partition_id]

    # Filter dataset for these users
    client_df = df[df["user"].isin(my_users)]
    print(f"Client {partition_id} has {len(client_df)} rows from users {list(my_users)}")

    # Extract features and labels
    X_raw = client_df[feature_cols].values
    y_raw = label_encoder.transform(client_df['class'].values)

    # Train/val split (before SMOTE to avoid data leakage)
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    # Apply SMOTE to balance training classes (with adaptive k_neighbors for small datasets)
    print(f"Client {partition_id} - Original train shape: {X_train.shape}")

    # Check minimum class size to set appropriate k_neighbors
    unique, counts = np.unique(y_train, return_counts=True)
    min_class_size = counts.min()

    if min_class_size < 2:
        # Can't use SMOTE with less than 2 samples in a class
        print(f"Client {partition_id} - Skipping SMOTE (min class size: {min_class_size})")
        X_train_resampled, y_train_resampled = X_train, y_train
    else:
        # Set k_neighbors to min(5, min_class_size - 1) to avoid errors
        k_neighbors = min(5, min_class_size - 1)
        print(f"Client {partition_id} - Using SMOTE with k_neighbors={k_neighbors}")
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Client {partition_id} - After SMOTE train shape: {X_train_resampled.shape}")

    # Scale features (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)

    return (X_train_scaled, y_train_resampled), (X_val_scaled, y_val)


def get_model_params(num_features: int, num_classes: int, hidden_size: int):
    """Get model info for initialization."""
    return num_features, num_classes, hidden_size


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model weights as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]):
    """Set model weights from list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(model: nn.Module, train_loader: DataLoader, config: Dict, device: torch.device,
          total_steps: int = 0) -> Tuple[float, int, float]:
    """
    Train model for one round with optional Differential Privacy.

    Features:
    - Optional FocalLoss for hard example mining
    - Class weights for imbalance handling
    - Label smoothing for regularization
    - Cosine annealing LR schedule

    Returns: (loss, num_samples, epsilon, total_steps)
    """
    model.train()

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4  # Reduced from 1e-3
    )

    # Learning rate scheduler (cosine annealing within local epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["local_epochs"] * len(train_loader),
        eta_min=config["learning_rate"] * 0.1
    )

    # Setup loss function
    class_weights = config.get("class_weights", None)
    use_focal_loss = config.get("use_focal_loss", False)  # Default to False for stability

    if use_focal_loss:
        alpha = torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
        criterion = FocalLoss(
            alpha=alpha,
            gamma=1.5,  # Reduced from 2.0 for more stability
            label_smoothing=0.05,  # Reduced from 0.1
            num_classes=config["num_classes"]
        )
    elif class_weights is not None:
        weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Setup DP optimizer if enabled
    dp_optimizer = None
    if config["enable_dp"]:
        dp_optimizer = DPOptimizer(
            optimizer,
            max_grad_norm=config["max_grad_norm"],
            noise_multiplier=config["noise_multiplier"]
        )

    total_loss = 0.0
    steps = 0

    for epoch in range(config["local_epochs"]):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            if dp_optimizer is not None:
                dp_optimizer.step(model, len(data))
            else:
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            steps += 1

    total_steps += steps

    # Compute epsilon
    epsilon = 0.0
    if config["enable_dp"]:
        epsilon = compute_epsilon(
            steps=total_steps,
            batch_size=config["batch_size"],
            dataset_size=len(train_loader.dataset),
            noise_multiplier=config["noise_multiplier"],
            delta=config["target_delta"]
        )

    avg_loss = total_loss / steps if steps > 0 else 0.0
    return avg_loss, len(train_loader.dataset), epsilon, total_steps


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float, int]:
    """Evaluate model on test data. Returns (loss, accuracy, num_samples)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * len(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return avg_loss, accuracy, total
