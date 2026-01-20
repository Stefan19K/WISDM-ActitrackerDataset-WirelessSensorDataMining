import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from wisdm_har_classification.logger import logger

WISDM_DATASET = "WISDM_ar_v1.1"
WISDM_DATASET_A = f"{WISDM_DATASET}/client_A.csv"

class WISDMDataset(Dataset):
    """
    Simple Dataset wrapper for Tabular Transformed Data.
    Input: Pre-processed numpy arrays X (features) and y (labels)
    """
    def __init__(self, X, y):
        # Convert to Torch Tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

def load_data(
    partition_id=0,
    num_partitions=1,
    path=WISDM_DATASET_A,
    batch_size=32
):
    """
    Loads the transformed feature CSV, filters by user based on partition_id,
    applies SMOTE, and returns DataLoaders.
    """
    logger.info(f"Loading data from: {path}")

    # 1. Load Data
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"File not found: {path}. Please run the EDA script to generate client files.")
        raise

    # -------------------------------------------------------------------------
    # PARTITIONING: Assign specific Users to this Client
    # -------------------------------------------------------------------------
    users = df["user"].unique()
    users.sort()

    # Split users among clients
    # e.g. if 36 users and 4 clients -> 9 users per client
    client_users_map = np.array_split(users, num_partitions)
    my_users = client_users_map[partition_id]

    # Filter dataset for these users
    df = df[df["user"].isin(my_users)]

    logger.info(f"Client {partition_id} assigned users: {my_users}")
    logger.info(f"Client {partition_id} has {len(df)} rows")

    # 2. Separate Features and Target
    # Identifying feature columns (everything except 'user' and 'activity')
    # Transformed data usually has columns like 'feat_0', 'feat_1'... or specific names.
    # We drop metadata columns.
    drop_cols = ['class', 'user', 'UNIQUE_ID']
    # Use errors='ignore' in case 'user' or 'Client_ID' aren't present
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], axis=1).values
    y_raw = df['class'].values

    # 3. Label Encoding (String -> Int)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    # Print mapping for sanity check
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Class Mapping: {mapping}")

    # 4. Train / Val Split (Shuffle is usually okay for Tabular data, unlike Time-Series)
    # We split BEFORE SMOTE to avoid data leakage
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    logger.info(f"Original Train shape: {X_train.shape}, Count: {len(y_train)}")

    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Train Distribution: {X_train.shape}, Class {unique[0]}: {counts[0]}, Class {unique[1]}: {counts[1]}, Class {unique[2]}: {counts[2]}, Class {unique[3]}: {counts[3]}, Class {unique[4]}: {counts[4]}, Class {unique[5]}: {counts[5]}")

    # 5. Apply SMOTE (Oversampling) - ONLY on Training Data
    # This solves your "fewer examples" problem by generating synthetic points
    logger.info("Applying SMOTE to balance training classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    logger.info(f"New SMOTE Train shape: {X_train_resampled.shape}, Count: {len(y_train_resampled)}")

    unique, counts = np.unique(y_train_resampled, return_counts=True)
    logger.info(f"New SMOTE Train Distribution: {X_train_resampled.shape}, Class {unique[0]}: {counts[0]}, Class {unique[1]}: {counts[1]}, Class {unique[2]}: {counts[2]}, Class {unique[3]}: {counts[3]}, Class {unique[4]}: {counts[4]}, Class {unique[5]}: {counts[5]}")

    # 6. Scaling (Standardization)
    # Fit scaler ONLY on training data, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)

    # 7. Create Datasets and Loaders
    train_dataset = WISDMDataset(X_train_scaled, y_train_resampled)
    val_dataset = WISDMDataset(X_val_scaled, y_val)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader

# Quick test block (Run this file directly to check)
if __name__ == "__main__":
    try:
        t_loader, v_loader = load_data()
        X_batch, y_batch = next(iter(t_loader))
        print(f"\nSuccess! Batch X shape: {X_batch.shape}") # Should be [32, 46] roughly
        print(f"Batch y shape: {y_batch.shape}")
    except Exception as e:
        print(f"Error: {e}")
