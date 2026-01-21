"""Cross-Validation Script for WISDM HAR Models.

Evaluates 4 models trained on different client datasets:
- Model A (PyTorch MLP): trained on client_A.csv (feature-extracted)
- Model B (PyTorch CNN with DP): trained on client_B.csv (feature-extracted)
- Model C (Keras CNN Robust): trained on client_C.csv (52 features, noise-augmented)
- Model D (Keras CNN FFT): trained on client_C.csv (52 features)

All models are evaluated on all 3 client datasets (feature-extracted).
Models C and D use a scaler fitted on client_C for consistent preprocessing.
Run from src as working dir.
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import zipfile
import tempfile
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------
# Models

class ModelA_MLP(nn.Module):
    """Model A: Simple MLP (Net from wisdm-har-classification)."""
    def __init__(self, input_dim=43, num_classes=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.model(x)


class ModelB_CNN(nn.Module):
    """Model B: 1D CNN with GroupNorm (ActivityCNN from wisdm-har-dp)."""
    def __init__(self, num_features=46, num_classes=6, hidden_size=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 128)
        self.fc1 = nn.Linear(128 * num_features, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ModelD_DeepCNN(nn.Module):
    """Model D: Deep 1D CNN for raw time-series with FFT features (from wisdm_classification_raw.ipynb).

    Architecture matches Keras model:
    - Conv Block 1: Conv1D(64) -> Conv1D(64) -> MaxPool(2) -> Dropout(0.3)
    - Conv Block 2: Conv1D(128) -> Conv1D(128) -> MaxPool(2) -> Dropout(0.3)
    - Conv Block 3: Conv1D(256) -> MaxPool(2) -> Dropout(0.4)
    - Dense: Flatten -> Dense(128) -> Dropout(0.5) -> Dense(6)

    Input: (batch, window_size=200, channels=6) where channels = [x,y,z,fft_x,fft_y,fft_z]
    """
    def __init__(self, window_size=200, n_channels=6, n_classes=6):
        super().__init__()
        # Conv Block 1
        self.conv1_1 = nn.Conv1d(n_channels, 64, kernel_size=3, padding=0)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)

        # Conv Block 2
        self.conv2_1 = nn.Conv1d(64, 128, kernel_size=3, padding=0)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)

        # Conv Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.4)

        # Calculate flatten size: input 200 -> 198 -> 196 -> 98 -> 96 -> 94 -> 47 -> 45 -> 22
        # 22 * 256 = 5632
        flatten_size = 256 * 22

        # Dense layers
        self.fc1 = nn.Linear(flatten_size, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # Input: (batch, window_size, channels) -> (batch, channels, window_size)
        x = x.transpose(1, 2)

        # Conv Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Conv Block 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten and Dense
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x


# ------------------------------------------------------------------------
# Feature engineering functions

def engineer_features_model_b(df, feature_cols):
    """Feature engineering for Model B (from task.py)."""
    df = df.copy()
    new_features = []
    epsilon = 1e-8

    # Identify axis-specific columns
    x_cols = [c for c in feature_cols if 'X' in c.upper()]
    y_cols = [c for c in feature_cols if 'Y' in c.upper()]
    z_cols = [c for c in feature_cols if 'Z' in c.upper()]

    # Find statistical columns
    x_mean = [c for c in x_cols if 'avg' in c.lower()]
    y_mean = [c for c in y_cols if 'avg' in c.lower()]
    z_mean = [c for c in z_cols if 'avg' in c.lower()]
    x_std = [c for c in x_cols if 'standdev' in c.lower()]
    y_std = [c for c in y_cols if 'standdev' in c.lower()]
    z_std = [c for c in z_cols if 'standdev' in c.lower()]

    # Vertical dominance ratio
    if y_std and x_std and z_std:
        y_var = df[y_std[0]]**2
        x_var = df[x_std[0]]**2
        z_var = df[z_std[0]]**2
        total_var = x_var + y_var + z_var + epsilon
        df['vertical_dominance'] = y_var / total_var
        df['horizontal_dominance'] = (x_var + z_var) / total_var
        new_features.extend(['vertical_dominance', 'horizontal_dominance'])

    # Y-axis mean features
    if y_mean:
        df['y_mean_sign'] = np.sign(df[y_mean[0]])
        new_features.append('y_mean_sign')

    return df, feature_cols + new_features


def engineer_features_model_c(df):
    """Feature engineering for Model C (from train_robust_model.py)."""
    df = df.copy()
    epsilon = 0.001

    required_cols = ['ZAVG', 'ZSTANDDEV', 'YSTANDDEV', 'RESULTANT', 'XSTANDDEV', 'ZABSOLDEV', 'YABSOLDEV']
    if not all(col in df.columns for col in required_cols):
        return df

    df['Z_direction'] = np.sign(df['ZAVG'])
    df['vertical_intensity'] = df['ZSTANDDEV'] / (df['YSTANDDEV'] + epsilon)
    df['Z_to_resultant'] = df['ZAVG'].abs() / (df['RESULTANT'] + epsilon)
    df['movement_stability'] = 1 / (df['XSTANDDEV'] + df['YSTANDDEV'] + df['ZSTANDDEV'] + epsilon)
    df['X_to_Y_std_ratio'] = df['XSTANDDEV'] / (df['YSTANDDEV'] + epsilon)
    df['vertical_deviation_ratio'] = df['ZABSOLDEV'] / (df['YABSOLDEV'] + epsilon)

    # Range features
    for axis in ['X', 'Y', 'Z']:
        cols = [f'{axis}{i}' for i in range(10)]
        if all(c in df.columns for c in cols):
            df[f'{axis}_range'] = df[cols].max(axis=1) - df[cols].min(axis=1)

    return df


# ------------------------------------------------------------------------
# Raw time-series data functions (for Model E)

def load_raw_wisdm_data(raw_data_path):
    """Load raw WISDM accelerometer data from text file.

    Args:
        raw_data_path: Path to WISDM_ar_v1.1_raw.txt

    Returns:
        DataFrame with columns: user_id, activity, timestamp, x_accel, y_accel, z_accel
    """
    data = []
    with open(raw_data_path, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')
            if not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) >= 6:
                    user_id = int(parts[0])
                    activity = parts[1]
                    timestamp = int(parts[2])
                    x_accel = float(parts[3])
                    y_accel = float(parts[4])
                    z_accel = float(parts[5].rstrip(';'))
                    data.append([user_id, activity, timestamp, x_accel, y_accel, z_accel])
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(data, columns=['user_id', 'activity', 'timestamp', 'x_accel', 'y_accel', 'z_accel'])
    return df


def create_windows(df, window_size=200, step_size=100):
    """Create sliding windows from continuous accelerometer data.

    Args:
        df: DataFrame with user_id, activity, x_accel, y_accel, z_accel
        window_size: Number of timesteps per window (default: 200 = 10 seconds at 20Hz)
        step_size: Step between windows (default: 100 = 50% overlap)

    Returns:
        windows: Array of shape (n_windows, window_size, 3)
        labels: Array of activity labels
        user_ids: Array of user IDs for each window
    """
    windows = []
    labels = []
    user_ids = []

    for (user, activity), group in df.groupby(['user_id', 'activity']):
        data = group[['x_accel', 'y_accel', 'z_accel']].values

        for i in range(0, len(data) - window_size, step_size):
            window = data[i:i + window_size]
            if window.shape[0] == window_size:
                windows.append(window)
                labels.append(activity)
                user_ids.append(user)

    return np.array(windows, dtype=np.float32), np.array(labels), np.array(user_ids)


def add_fft_features(windows):
    """Compute FFT magnitude spectrum and concatenate with time-domain data.

    Args:
        windows: Array of shape (n_windows, window_size, 3) - time domain data

    Returns:
        combined: Array of shape (n_windows, window_size, 6) where:
                 - channels 0-2: time domain (x, y, z)
                 - channels 3-5: FFT magnitude (x, y, z)
    """
    n_samples, window_size, n_channels = windows.shape
    fft_features = np.zeros_like(windows)

    for i in range(n_samples):
        for axis in range(n_channels):
            fft_result = np.fft.fft(windows[i, :, axis])
            fft_magnitude = np.abs(fft_result[:window_size])
            fft_features[i, :, axis] = fft_magnitude

    return np.concatenate([windows, fft_features], axis=2)


def get_client_users(client_csv_path):
    """Get list of user IDs from a client CSV file."""
    df = pd.read_csv(client_csv_path)
    return df['user'].unique().tolist()


def load_raw_data_for_client(raw_data_path, client_users, window_size=200, step_size=100):
    """Load and preprocess raw data for specific client users.

    Args:
        raw_data_path: Path to WISDM_ar_v1.1_raw.txt
        client_users: List of user IDs for this client
        window_size: Window size for time-series
        step_size: Step size for sliding window

    Returns:
        X: Array of shape (n_windows, window_size, 6)
        y: Encoded labels
        le: LabelEncoder
    """
    df = load_raw_wisdm_data(raw_data_path)
    df = df[df['user_id'].isin(client_users)]

    windows, labels, _ = create_windows(df, window_size=window_size, step_size=step_size)
    X_with_fft = add_fft_features(windows)

    le = LabelEncoder()
    le.fit(['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'])
    y_encoded = le.transform(labels)

    return X_with_fft, y_encoded, le


# ------------------------------------------------------------------------
# Data loading functions

def load_data_for_model_a(csv_path):
    """Load and preprocess data for Model A (43 features, no engineering)."""
    df = pd.read_csv(csv_path)

    drop_cols = ['class', 'user', 'UNIQUE_ID']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df['class'].values

    le = LabelEncoder()
    le.fit(['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'])
    y_encoded = le.transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, le


def load_data_for_model_b(csv_path):
    """Load and preprocess data for Model B (46 features with engineering)."""
    df = pd.read_csv(csv_path)

    drop_cols = ['class', 'user', 'UNIQUE_ID']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Apply feature engineering
    df, feature_cols = engineer_features_model_b(df, feature_cols)

    X = df[feature_cols].values
    y = df['class'].values

    le = LabelEncoder()
    le.fit(['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'])
    y_encoded = le.transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, le


def load_data_for_model_c(csv_path, scaler=None, return_scaler=False):
    """Load and preprocess data for Model C (52 features with engineering).

    Args:
        csv_path: Path to client CSV file
        scaler: Pre-fitted StandardScaler. If None, fits a new one on this data.
        return_scaler: If True, also return the scaler (for fitting on reference data)

    Returns:
        X_scaled, y_encoded, le, [scaler if return_scaler]
    """
    df = pd.read_csv(csv_path)

    # Apply feature engineering
    df = engineer_features_model_c(df)

    non_feature_cols = ['UNIQUE_ID', 'user', 'class']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = df[feature_cols].values
    y = df['class'].values

    le = LabelEncoder()
    le.fit(['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking'])
    y_encoded = le.transform(y)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if return_scaler:
        return X_scaled, y_encoded, le, scaler
    return X_scaled, y_encoded, le


# ------------------------------------------------------------------------
# Model loading functions

def load_model_a(weights_path):
    """Load Model A (PyTorch MLP) from numpy weights."""
    model = ModelA_MLP(input_dim=43, num_classes=6)
    weights = np.load(weights_path, allow_pickle=True)

    state_dict = model.state_dict()
    for i, (key, _) in enumerate(state_dict.items()):
        state_dict[key] = torch.tensor(weights[i])

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_b(weights_path):
    """Load Model B (PyTorch CNN) from numpy weights.

    The DP model saves weights as a list of numpy arrays (via get_weights in task.py),
    so we need to match them with the model's state_dict keys.
    """
    model = ModelB_CNN(num_features=46, num_classes=6, hidden_size=64)
    weights_data = np.load(weights_path, allow_pickle=True)

    # Check if it's a dict (saved via .item()) or a list/array
    if isinstance(weights_data, np.ndarray) and weights_data.ndim == 0:
        # It's a 0-d array containing a dict
        weights_dict = weights_data.item()
        state_dict = {}
        for key, value in weights_dict.items():
            state_dict[key] = torch.tensor(value)
    else:
        # It's a list of weight arrays - match with state_dict keys
        state_dict = model.state_dict()
        for i, (key, _) in enumerate(state_dict.items()):
            state_dict[key] = torch.tensor(weights_data[i])

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_c_weights(weights_path):
    """Load Model C weights from Keras file (supports both HDF5 and Keras 3 zip format)."""
    weights = {}

    # First, try HDF5 format (Keras 2.x / TF SavedModel)
    try:
        with h5py.File(weights_path, 'r') as f:
            # Check if it's the expected structure
            if 'model_weights' in f:
                weights['conv1_kernel'] = f['model_weights/conv1d/conv1d/kernel:0'][:]
                weights['conv1_bias'] = f['model_weights/conv1d/conv1d/bias:0'][:]
                weights['conv2_kernel'] = f['model_weights/conv1d_1/conv1d_1/kernel:0'][:]
                weights['conv2_bias'] = f['model_weights/conv1d_1/conv1d_1/bias:0'][:]
                weights['dense1_kernel'] = f['model_weights/dense/dense/kernel:0'][:]
                weights['dense1_bias'] = f['model_weights/dense/dense/bias:0'][:]
                weights['dense2_kernel'] = f['model_weights/dense_1/dense_1/kernel:0'][:]
                weights['dense2_bias'] = f['model_weights/dense_1/dense_1/bias:0'][:]
                return weights
    except Exception as e:
        print(f"  Note: Could not load as HDF5 ({e}), trying Keras 3 zip format...")

    # Try Keras 3 zip format
    try:
        with zipfile.ZipFile(weights_path, 'r') as zf:
            # Keras 3 stores weights in model.weights.h5 inside the zip
            with zf.open('model.weights.h5') as weights_file:
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                    tmp.write(weights_file.read())
                    tmp_path = tmp.name

                with h5py.File(tmp_path, 'r') as f:
                    # Keras 3 structure: layers/<layer_name>/vars/0, vars/1, etc.
                    # vars/0 is typically kernel, vars/1 is bias

                    # Conv1D layer
                    weights['conv1_kernel'] = f['layers/conv1d/vars/0'][:]
                    weights['conv1_bias'] = f['layers/conv1d/vars/1'][:]

                    # Conv1D_1 layer
                    weights['conv2_kernel'] = f['layers/conv1d_1/vars/0'][:]
                    weights['conv2_bias'] = f['layers/conv1d_1/vars/1'][:]

                    # Dense layer
                    weights['dense1_kernel'] = f['layers/dense/vars/0'][:]
                    weights['dense1_bias'] = f['layers/dense/vars/1'][:]

                    # Dense_1 layer
                    weights['dense2_kernel'] = f['layers/dense_1/vars/0'][:]
                    weights['dense2_bias'] = f['layers/dense_1/vars/1'][:]

                os.unlink(tmp_path)
                return weights
    except Exception as e:
        print(f"  Note: Could not load as Keras 3 zip ({e})")

    raise ValueError(f"Could not load weights from {weights_path}. Please ensure the file is a valid Keras model file.")


class ModelC_CNN_PyTorch(nn.Module):
    """Model C converted to PyTorch for unified evaluation (cnn_model from wisdm-har-robustness/wisdm_classification.ipynb)."""
    def __init__(self, n_features=52, n_classes=6):
        super().__init__()
        # Match Keras architecture: Conv1D(64) -> Dropout -> Conv1D(64) -> MaxPool -> Flatten -> Dense(100) -> Dropout -> Dense(6)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=0)  # valid padding
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # Calculate flatten size: (n_features - 2 - 2) // 2 * 64
        flatten_size = ((n_features - 2 - 2) // 2) * 64
        self.fc1 = nn.Linear(flatten_size, 100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x):
        # Input: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def load_model_c(weights_path):
    """Load Model C (Keras CNN Robust, converted to PyTorch)."""
    weights = load_model_c_weights(weights_path)
    model = ModelC_CNN_PyTorch(n_features=52, n_classes=6)

    # Convert Keras weights to PyTorch format
    # Keras Conv1D: (kernel_size, in_channels, out_channels) -> PyTorch: (out_channels, in_channels, kernel_size)
    state_dict = model.state_dict()
    state_dict['conv1.weight'] = torch.tensor(weights['conv1_kernel'].transpose(2, 1, 0))
    state_dict['conv1.bias'] = torch.tensor(weights['conv1_bias'])
    state_dict['conv2.weight'] = torch.tensor(weights['conv2_kernel'].transpose(2, 1, 0))
    state_dict['conv2.bias'] = torch.tensor(weights['conv2_bias'])
    # Keras Dense: (in_features, out_features) -> PyTorch: (out_features, in_features)
    state_dict['fc1.weight'] = torch.tensor(weights['dense1_kernel'].T)
    state_dict['fc1.bias'] = torch.tensor(weights['dense1_bias'])
    state_dict['fc2.weight'] = torch.tensor(weights['dense2_kernel'].T)
    state_dict['fc2.bias'] = torch.tensor(weights['dense2_bias'])

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_d_weights(weights_path):
    """Load Model D weights from Keras file (deep CNN for raw time-series).

    Model D has 5 conv layers: Conv1D(64) x2, Conv1D(128) x2, Conv1D(256).
    """
    weights = {}

    # Try HDF5 format first (Keras 2.x)
    try:
        with h5py.File(weights_path, 'r') as f:
            if 'model_weights' in f:
                # Conv Block 1
                weights['conv1_1_kernel'] = f['model_weights/conv1d/conv1d/kernel:0'][:]
                weights['conv1_1_bias'] = f['model_weights/conv1d/conv1d/bias:0'][:]
                weights['conv1_2_kernel'] = f['model_weights/conv1d_1/conv1d_1/kernel:0'][:]
                weights['conv1_2_bias'] = f['model_weights/conv1d_1/conv1d_1/bias:0'][:]
                # Conv Block 2
                weights['conv2_1_kernel'] = f['model_weights/conv1d_2/conv1d_2/kernel:0'][:]
                weights['conv2_1_bias'] = f['model_weights/conv1d_2/conv1d_2/bias:0'][:]
                weights['conv2_2_kernel'] = f['model_weights/conv1d_3/conv1d_3/kernel:0'][:]
                weights['conv2_2_bias'] = f['model_weights/conv1d_3/conv1d_3/bias:0'][:]
                # Conv Block 3
                weights['conv3_kernel'] = f['model_weights/conv1d_4/conv1d_4/kernel:0'][:]
                weights['conv3_bias'] = f['model_weights/conv1d_4/conv1d_4/bias:0'][:]
                # Dense
                weights['dense1_kernel'] = f['model_weights/dense/dense/kernel:0'][:]
                weights['dense1_bias'] = f['model_weights/dense/dense/bias:0'][:]
                weights['dense2_kernel'] = f['model_weights/dense_1/dense_1/kernel:0'][:]
                weights['dense2_bias'] = f['model_weights/dense_1/dense_1/bias:0'][:]
                return weights
    except Exception as e:
        print(f"  Note: Could not load Model D as HDF5 ({e}), trying Keras 3 zip format...")

    # Try Keras 3 zip format
    try:
        with zipfile.ZipFile(weights_path, 'r') as zf:
            with zf.open('model.weights.h5') as weights_file:
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                    tmp.write(weights_file.read())
                    tmp_path = tmp.name

                with h5py.File(tmp_path, 'r') as f:
                    # Conv Block 1
                    weights['conv1_1_kernel'] = f['layers/conv1d/vars/0'][:]
                    weights['conv1_1_bias'] = f['layers/conv1d/vars/1'][:]
                    weights['conv1_2_kernel'] = f['layers/conv1d_1/vars/0'][:]
                    weights['conv1_2_bias'] = f['layers/conv1d_1/vars/1'][:]
                    # Conv Block 2
                    weights['conv2_1_kernel'] = f['layers/conv1d_2/vars/0'][:]
                    weights['conv2_1_bias'] = f['layers/conv1d_2/vars/1'][:]
                    weights['conv2_2_kernel'] = f['layers/conv1d_3/vars/0'][:]
                    weights['conv2_2_bias'] = f['layers/conv1d_3/vars/1'][:]
                    # Conv Block 3
                    weights['conv3_kernel'] = f['layers/conv1d_4/vars/0'][:]
                    weights['conv3_bias'] = f['layers/conv1d_4/vars/1'][:]
                    # Dense
                    weights['dense1_kernel'] = f['layers/dense/vars/0'][:]
                    weights['dense1_bias'] = f['layers/dense/vars/1'][:]
                    weights['dense2_kernel'] = f['layers/dense_1/vars/0'][:]
                    weights['dense2_bias'] = f['layers/dense_1/vars/1'][:]

                os.unlink(tmp_path)
                return weights
    except Exception as e:
        print(f"  Note: Could not load Model D as Keras 3 zip ({e})")

    raise ValueError(f"Could not load Model D weights from {weights_path}")


def load_model_d(weights_path):
    """Load Model D (Deep CNN for raw time-series, converted to PyTorch)."""
    weights = load_model_d_weights(weights_path)
    model = ModelD_DeepCNN(window_size=200, n_channels=6, n_classes=6)

    # Convert Keras weights to PyTorch format
    # Keras Conv1D: (kernel_size, in_channels, out_channels) -> PyTorch: (out_channels, in_channels, kernel_size)
    state_dict = model.state_dict()

    # Conv Block 1
    state_dict['conv1_1.weight'] = torch.tensor(weights['conv1_1_kernel'].transpose(2, 1, 0))
    state_dict['conv1_1.bias'] = torch.tensor(weights['conv1_1_bias'])
    state_dict['conv1_2.weight'] = torch.tensor(weights['conv1_2_kernel'].transpose(2, 1, 0))
    state_dict['conv1_2.bias'] = torch.tensor(weights['conv1_2_bias'])

    # Conv Block 2
    state_dict['conv2_1.weight'] = torch.tensor(weights['conv2_1_kernel'].transpose(2, 1, 0))
    state_dict['conv2_1.bias'] = torch.tensor(weights['conv2_1_bias'])
    state_dict['conv2_2.weight'] = torch.tensor(weights['conv2_2_kernel'].transpose(2, 1, 0))
    state_dict['conv2_2.bias'] = torch.tensor(weights['conv2_2_bias'])

    # Conv Block 3
    state_dict['conv3.weight'] = torch.tensor(weights['conv3_kernel'].transpose(2, 1, 0))
    state_dict['conv3.bias'] = torch.tensor(weights['conv3_bias'])

    # Dense layers
    state_dict['fc1.weight'] = torch.tensor(weights['dense1_kernel'].T)
    state_dict['fc1.bias'] = torch.tensor(weights['dense1_bias'])
    state_dict['fc2.weight'] = torch.tensor(weights['dense2_kernel'].T)
    state_dict['fc2.bias'] = torch.tensor(weights['dense2_bias'])

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ------------------------------------------------------------------------
# Evaluation Funstions

def evaluate_pytorch_model(model, X, y, device='cpu'):
    """Evaluate a PyTorch model."""
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)

    y_pred = predictions.cpu().numpy()

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'f1_weighted': f1_score(y, y_pred, average='weighted'),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro'),
    }

    return metrics, y_pred


def run_cross_validation(data_dir, models_dir, model_c_file='wisdm_cnn_robust_model.keras',
                         model_d_file='wisdm_cnn_fft_model.keras'):
    """Run full cross-validation across all models and datasets.

    Args:
        data_dir: Directory containing client_A/B/C.csv files
        models_dir: Directory containing wisdm-har-* model folders
        model_c_file: Filename for Model C (robust CNN)
        model_d_file: Filename for Model D (FFT CNN)
    """

    # File paths
    data_files = {
        'A': os.path.join(data_dir, 'client_A.csv'),
        'B': os.path.join(data_dir, 'client_B.csv'),
        'C': os.path.join(data_dir, 'client_C.csv'),
    }

    model_files = {
        'A': os.path.join(models_dir, 'wisdm-har-classification/save_params/final_global_model.npy'),
        'B': os.path.join(models_dir, 'wisdm-har-dp/save_params/federated_dp_model.npy'),
        'C': os.path.join(models_dir, 'wisdm-har-robustness', model_c_file),
        'D': os.path.join(models_dir, 'wisdm-har-robustness', model_d_file),
    }

    # Load models
    print("Loading models...")
    print(f"Loading Model A from: {model_files['A']}")
    models = {
        'A': load_model_a(model_files['A']),
    }
    print("Model A loaded")

    print(f"Loading Model B from: {model_files['B']}")
    models['B'] = load_model_b(model_files['B'])
    print("Model B loaded")

    print(f"Loading Model C from: {model_files['C']}")
    model_c_path = model_files['C']
    file_size = os.path.getsize(model_c_path)
    with open(model_c_path, 'rb') as f:
        header = f.read(8)
    print(f"    File size: {file_size} bytes")
    is_hdf5 = header[:4] == b'\x89HDF'
    is_zip = header[:4] == b'PK\x03\x04'
    print(f"    Format detected: {'HDF5' if is_hdf5 else 'ZIP' if is_zip else 'Unknown'}")
    models['C'] = load_model_c(model_c_path)
    print("Model C loaded")

    # Model D has same architecture as Model C (2-layer CNN on 52 features)
    print(f"Loading Model D from: {model_files['D']}")
    model_d_path = model_files['D']
    file_size = os.path.getsize(model_d_path)
    with open(model_d_path, 'rb') as f:
        header = f.read(8)
    print(f"    File size: {file_size} bytes")
    is_hdf5 = header[:4] == b'\x89HDF'
    is_zip = header[:4] == b'PK\x03\x04'
    print(f"    Format detected: {'HDF5' if is_hdf5 else 'ZIP' if is_zip else 'Unknown'}")
    # Use load_model_c since Model D has same architecture
    models['D'] = load_model_c(model_d_path)
    print("Model D loaded")
    print("All models loaded\n")

    # Training source labels
    train_sources = {
        'A': 'client_A',
        'B': 'client_B',
        'C': 'client_C (robust)',
        'D': 'client_C (fft)',
    }

    # Results storage
    results = {}
    all_predictions = {}

    # First, fit a scaler on client_C for Models C and D
    print("Fitting scaler on client_C for Models C and D...")
    _, _, _, scaler_c = load_data_for_model_c(data_files['C'], scaler=None, return_scaler=True)
    print("Scaler fitted\n")

    # Cross-validation for Models A and B (each uses its own scaling)
    print("Cross-validation results:")

    for model_name in ['A', 'B']:
        model = models[model_name]
        results[model_name] = {}
        all_predictions[model_name] = {}

        print(f"\n--- Model {model_name} ---")
        print(f"(Trained on {train_sources[model_name]})")

        if model_name == 'A':
            loader = load_data_for_model_a
        else:
            loader = load_data_for_model_b

        for data_name in ['A', 'B', 'C']:
            X, y, le = loader(data_files[data_name])
            metrics, y_pred = evaluate_pytorch_model(model, X, y)
            results[model_name][data_name] = metrics
            all_predictions[model_name][data_name] = {'y_true': y, 'y_pred': y_pred}

            in_domain = " (in-domain)" if model_name == data_name else ""
            print(f"  Eval on client_{data_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}{in_domain}")

    # Cross-validation for Models C and D (use scaler fitted on client_C)
    for model_name in ['C', 'D']:
        model = models[model_name]
        results[model_name] = {}
        all_predictions[model_name] = {}

        print(f"\n--- Model {model_name} ---")
        print(f"(Trained on {train_sources[model_name]})")

        for data_name in ['A', 'B', 'C']:
            # Use the pre-fitted scaler from client_C
            X, y, le = load_data_for_model_c(data_files[data_name], scaler=scaler_c)
            metrics, y_pred = evaluate_pytorch_model(model, X, y)
            results[model_name][data_name] = metrics
            all_predictions[model_name][data_name] = {'y_true': y, 'y_pred': y_pred}

            # Both C and D were trained on client_C
            in_domain = " (in-domain)" if data_name == 'C' else ""
            print(f"  Eval on client_{data_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}{in_domain}")

    return results, all_predictions, le


def create_results_table(results):
    """Create a summary table of results."""
    model_names = list(results.keys())
    data_names = ['A', 'B', 'C']

    # Accuracy matrix
    acc_matrix = pd.DataFrame(
        index=[f'Model {m}' for m in model_names],
        columns=[f'Client {d}' for d in data_names]
    )

    f1_matrix = pd.DataFrame(
        index=[f'Model {m}' for m in model_names],
        columns=[f'Client {d}' for d in data_names]
    )

    for i, model_name in enumerate(model_names):
        for j, data_name in enumerate(data_names):
            acc_matrix.iloc[i, j] = results[model_name][data_name]['accuracy']
            f1_matrix.iloc[i, j] = results[model_name][data_name]['f1_macro']

    return acc_matrix, f1_matrix


def plot_results(results, output_dir):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results.keys())
    n_models = len(model_names)

    # Model display names
    model_labels = {
        'A': 'Model A\n(MLP)',
        'B': 'Model B\n(CNN+DP)',
        'C': 'Model C\n(Robust CNN)',
        'D': 'Model D\n(FFT CNN)',
    }

    # Create accuracy heatmap
    acc_matrix, f1_matrix = create_results_table(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5 + 0.5 * (n_models - 3)))

    # Accuracy heatmap
    acc_values = acc_matrix.astype(float).values
    sns.heatmap(acc_values, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=['Client A', 'Client B', 'Client C'],
                yticklabels=[model_labels.get(m, f'Model {m}') for m in model_names],
                ax=axes[0], vmin=0.3, vmax=1.0)
    axes[0].set_title('Cross-Validation Accuracy\n(Rows: Models, Cols: Test Data)', fontsize=12)
    axes[0].set_xlabel('Evaluation Dataset')
    axes[0].set_ylabel('Model (Trained on)')

    # F1 heatmap
    f1_values = f1_matrix.astype(float).values
    sns.heatmap(f1_values, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=['Client A', 'Client B', 'Client C'],
                yticklabels=[model_labels.get(m, f'Model {m}') for m in model_names],
                ax=axes[1], vmin=0.3, vmax=1.0)
    axes[1].set_title('Cross-Validation F1-Score (Macro)\n(Rows: Models, Cols: Test Data)', fontsize=12)
    axes[1].set_xlabel('Evaluation Dataset')
    axes[1].set_ylabel('Model (Trained on)')

    # Add diagonal annotation (in-domain) for A, B
    # C and D are both trained on client_C, so highlight column C for both
    for ax in axes:
        for i, m in enumerate(model_names):
            if m in ['A', 'B']:
                j = ['A', 'B', 'C'].index(m)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))
            elif m in ['C', 'D']:
                # Both C and D trained on client_C
                ax.add_patch(plt.Rectangle((2, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Create bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(3)
    width = 0.8 / n_models

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, model_name in enumerate(model_names):
        accuracies = [results[model_name][d]['accuracy'] for d in ['A', 'B', 'C']]
        bars = ax.bar(x + i*width - (n_models-1)*width/2, accuracies, width,
                      label=f'Model {model_name}', color=colors[i % len(colors)])

        # Highlight in-domain (A->A, B->B, C->C, D->C)
        if model_name in ['A', 'B']:
            in_domain_idx = ['A', 'B', 'C'].index(model_name)
            bars[in_domain_idx].set_edgecolor('black')
            bars[in_domain_idx].set_linewidth(2)
        elif model_name in ['C', 'D']:
            # Both C and D trained on client_C
            bars[2].set_edgecolor('black')
            bars[2].set_linewidth(2)

    ax.set_xlabel('Test Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Validation: Model Performance Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(['Client A', 'Client B', 'Client C'])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def save_detailed_results(results, all_predictions, le, output_dir):
    """Save detailed results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results.keys())
    data_names = ['A', 'B', 'C']

    # Training source mapping
    train_sources = {
        'A': 'Client A',
        'B': 'Client B',
        'C': 'Client C (robust)',
        'D': 'Client C (fft)',
    }

    # Summary CSV
    rows = []
    for model_name in model_names:
        for data_name in data_names:
            metrics = results[model_name][data_name]
            # In-domain check: A->A, B->B, C->C, D->C
            if model_name in ['C', 'D']:
                in_domain = data_name == 'C'
            else:
                in_domain = model_name == data_name
            rows.append({
                'Model': f'Model {model_name}',
                'Trained_On': train_sources.get(model_name, f'Client {model_name}'),
                'Evaluated_On': f'Client {data_name}',
                'In_Domain': in_domain,
                'Accuracy': metrics['accuracy'],
                'F1_Macro': metrics['f1_macro'],
                'F1_Weighted': metrics['f1_weighted'],
                'Precision_Macro': metrics['precision_macro'],
                'Recall_Macro': metrics['recall_macro'],
            })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)

    # Per-class metrics for each combination
    class_names = le.classes_

    for model_name in model_names:
        for data_name in data_names:
            y_true = all_predictions[model_name][data_name]['y_true']
            y_pred = all_predictions[model_name][data_name]['y_pred']

            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            df_report.to_csv(os.path.join(output_dir, f'report_model{model_name}_on_client{data_name}.csv'))

    print(f"Detailed results saved to {output_dir}/")

    return df_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Cross-validation for WISDM HAR models')
    parser.add_argument('--data-dir', default='../WISDM_ar_v1.1',
                        help='Directory containing client_A/B/C.csv files')
    parser.add_argument('--models-dir', default='.',
                        help='Directory containing wisdm-har-* model folders')
    parser.add_argument('--output-dir', default='./cross_validation_results',
                        help='Directory to save results')
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODELS_DIR = args.models_dir
    OUTPUT_DIR = args.output_dir
    MODEL_C_FILE = 'wisdm_cnn_robust_model.keras'
    MODEL_D_FILE = 'wisdm_cnn_fft_model.keras'

    # Verify paths exist
    print("Checking paths...")
    required_files = [
        os.path.join(DATA_DIR, 'client_A.csv'),
        os.path.join(DATA_DIR, 'client_B.csv'),
        os.path.join(DATA_DIR, 'client_C.csv'),
        os.path.join(MODELS_DIR, 'wisdm-har-classification/save_params/final_global_model.npy'),
        os.path.join(MODELS_DIR, 'wisdm-har-dp/save_params/federated_dp_model.npy'),
        os.path.join(MODELS_DIR, 'wisdm-har-robustness', MODEL_C_FILE),
        os.path.join(MODELS_DIR, 'wisdm-har-robustness', MODEL_D_FILE),
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease check your directory structure and paths.")
        print("Use --data-dir and --models-dir to specify custom paths.")
        exit(1)

    print("All required files found\n")

    # Run cross-validation
    results, all_predictions, le = run_cross_validation(
        DATA_DIR, MODELS_DIR, MODEL_C_FILE, MODEL_D_FILE
    )

    # Create summary tables
    print("\nSummary tables:")

    acc_matrix, f1_matrix = create_results_table(results)

    print("\nAccuracy Matrix:")
    print(acc_matrix.to_string())

    print("\nF1-Score (Macro) Matrix:")
    print(f1_matrix.to_string())

    # Calculate cross-domain performance drop
    print("\nGeneralization analysis:")

    for model_name in results.keys():
        # Models C and D are both trained on client_C
        if model_name in ['C', 'D']:
            in_domain_data = 'C'
        else:
            in_domain_data = model_name

        in_domain = results[model_name][in_domain_data]['accuracy']
        cross_domain = np.mean([results[model_name][d]['accuracy']
                               for d in ['A', 'B', 'C'] if d != in_domain_data])
        drop = in_domain - cross_domain
        print(f"Model {model_name}: In-domain={in_domain:.4f}, Cross-domain avg={cross_domain:.4f}, Drop={drop:.4f}")

    # Generate plots and save results
    plot_results(results, OUTPUT_DIR)
    df_results = save_detailed_results(results, all_predictions, le, OUTPUT_DIR)
