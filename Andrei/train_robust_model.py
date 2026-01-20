
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Helper Functions ---

def engineer_features(df):
    """Replicates the feature engineering from the original notebook."""
    df = df.copy()
    epsilon = 0.001
    
    # Check if columns exist before computing (to handle potential pre-processed subsets)
    required_cols = ['ZAVG', 'ZSTANDDEV', 'YSTANDDEV', 'RESULTANT', 'XSTANDDEV', 'ZABSOLDEV', 'YABSOLDEV']
    if not all(col in df.columns for col in required_cols):
        # Assuming if missing, we might already have features or different structure.
        # But based on client_C.csv structure, these should be there.
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

def add_noise_to_array(X, noise_level=0.1, seed=None):
    """Adds Gaussian noise to a numpy array."""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    return X + noise

# --- Main Script ---

def train_robust(data_path, output_model_path='wisdm_cnn_robust_model.keras', noise_levels=[0.05, 0.1]):
    print(f"Loading clean data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    non_feature_cols = ['UNIQUE_ID', 'user', 'class']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    X = df[feature_cols].values
    y = df['class'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)
    
    # Split Data (Stratified)
    print("Splitting data into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # --- Data Augmentation ---
    print(f"Augmenting training data with noise levels: {noise_levels}")
    X_train_augmented = [X_train]
    y_train_augmented = [y_train]
    
    for i, nl in enumerate(noise_levels):
        print(f"  Generating noise {nl}...")
        # Add noise to X_train
        X_noisy = add_noise_to_array(X_train, noise_level=nl, seed=42+i)
        X_train_augmented.append(X_noisy)
        y_train_augmented.append(y_train) # Labels stay the same
        
    X_train_combined = np.concatenate(X_train_augmented, axis=0)
    y_train_combined = np.concatenate(y_train_augmented, axis=0)
    
    print(f"Original Train size: {X_train.shape[0]}")
    print(f"Augmented Train size: {X_train_combined.shape[0]}")
    
    # Scale Data
    # Important: Fit scaler ONLY on the original clean training data to preserve real-world stats
    print("Scaling data (fit on clean train)...")
    scaler = StandardScaler()
    scaler.fit(X_train) 
    
    X_train_scaled = scaler.transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE
    # Apply SMOTE to the augmented dataset to handle class imbalance
    print("Applying SMOTE to augmented data...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_combined)
    
    # Reshape for CNN [samples, features, 1]
    X_train_cnn = np.expand_dims(X_train_resampled, axis=-1)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)
    
    y_train_cnn = to_categorical(y_train_resampled, num_classes=n_classes)
    y_test_cnn = to_categorical(y_test, num_classes=n_classes)
    
    # Build Model
    n_features = X_train_cnn.shape[1]
    
    model = Sequential([
        tf.keras.layers.Input(shape=(n_features, 1)),
        # Slightly increased dropout for better regularization on noisy data
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Dropout(0.4), 
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    print("Training robust model...")
    model.fit(
        X_train_cnn,
        y_train_cnn,
        epochs=20,
        batch_size=64, # Larger batch size for larger dataset
        validation_data=(X_test_cnn, y_test_cnn),
        verbose=1
    )
    
    # Save
    print(f"Saving robust model to {output_model_path}...")
    model.save(output_model_path)
    
    # Quick Eval on Clean Test
    y_pred_clean = np.argmax(model.predict(X_test_cnn), axis=1)
    acc_clean = accuracy_score(y_test, y_pred_clean)
    print(f"Accuracy on Clean Test Set: {acc_clean*100:.2f}%")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../WISDM_ar_v1.1/client_C.csv')
    parser.add_argument('--output', default='wisdm_cnn_robust_model.keras')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        # Try local if ../ fails
        if os.path.exists('client_C.csv'):
            args.input = 'client_C.csv'
        else:
            print(f"Error: Input file {args.input} not found.")
            exit(1)
            
    train_robust(args.input, args.output)
