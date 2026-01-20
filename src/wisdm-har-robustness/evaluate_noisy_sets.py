"""
evaluate_noisy_sets.py

Evaluate a saved Keras CNN model on the original `client_C.csv` and on generated noisy variants,
then plot accuracy differences.

Usage:
    python evaluate_noisy_sets.py --model-path wisdm_cnn_fft_model.keras --noisy-dir noisy_sets --original ../WISDM_ar_v1.1/client_C.csv --out-dir eval_results

If `--noisy-files` is provided, only those files will be evaluated (space separated).
"""

import os
import argparse
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

import tensorflow as tf
import sys


# Copy of feature engineering used in the notebook
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Z_direction'] = np.sign(df['ZAVG'])
    df['vertical_intensity'] = df['ZSTANDDEV'] / (df['YSTANDDEV'] + 0.001)
    df['Z_to_resultant'] = df['ZAVG'].abs() / (df['RESULTANT'] + 0.001)
    df['movement_stability'] = 1 / (df['XSTANDDEV'] + df['YSTANDDEV'] + df['ZSTANDDEV'] + 0.001)
    df['X_to_Y_std_ratio'] = df['XSTANDDEV'] / (df['YSTANDDEV'] + 0.001)
    df['vertical_deviation_ratio'] = df['ZABSOLDEV'] / (df['YABSOLDEV'] + 0.001)
    z_cols = [f'Z{i}' for i in range(10)]
    x_cols = [f'X{i}' for i in range(10)]
    y_cols = [f'Y{i}' for i in range(10)]
    df['Z_range'] = df[z_cols].max(axis=1) - df[z_cols].min(axis=1)
    df['X_range'] = df[x_cols].max(axis=1) - df[x_cols].min(axis=1)
    df['Y_range'] = df[y_cols].max(axis=1) - df[y_cols].min(axis=1)
    return df


def load_and_prepare(path: str, feature_cols: List[str], scaler: StandardScaler, label_enc: LabelEncoder):
    df = pd.read_csv(path)
    df = engineer_features(df)
    X = df[feature_cols].values
    y = df['class'].values
    y_enc = label_enc.transform(y)
    X_scaled = scaler.transform(X)
    # reshape for Conv1D: [samples, timesteps, channels]
    X_cnn = np.expand_dims(X_scaled, axis=-1)
    return X_cnn, y_enc, df


def evaluate_model(model_path: str, original_csv: str, noisy_dir: str, noisy_files: Optional[List[str]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Load original data to rebuild preprocessing and label mapping
    df_orig = pd.read_csv(original_csv)
    df_orig = engineer_features(df_orig)

    non_feature_cols = ['UNIQUE_ID', 'user', 'class']
    feature_cols = [c for c in df_orig.columns if c not in non_feature_cols]

    # Fit label encoder on original classes (replicates notebook behaviour)
    label_enc = LabelEncoder()
    label_enc.fit(df_orig['class'].values)

    # Fit scaler on original features (replicates notebook behaviour)
    scaler = StandardScaler()
    X_orig = df_orig[feature_cols].values
    scaler.fit(X_orig)

    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"ERROR: could not load model '{model_path}': {e}")
        print("Aborting evaluation. Provide a valid saved Keras model via --model-path.")
        sys.exit(1)

    # Evaluate clean baseline
    X_clean_cnn = np.expand_dims(scaler.transform(X_orig), axis=-1)
    y_clean = label_enc.transform(df_orig['class'].values)
    y_pred_clean = np.argmax(model.predict(X_clean_cnn), axis=1)
    acc_clean = accuracy_score(y_clean, y_pred_clean)
    bal_clean = balanced_accuracy_score(y_clean, y_pred_clean)
    f1_clean = f1_score(y_clean, y_pred_clean, average='macro')

    results = [{'label': 'clean', 'file': os.path.basename(original_csv), 'accuracy': acc_clean, 'balanced_accuracy': bal_clean, 'f1_macro': f1_clean}]

    # Determine noisy files
    if noisy_files:
        files = noisy_files
    else:
        files = sorted([f for f in os.listdir(noisy_dir) if f.startswith('client_C_noise') and f.endswith('.csv')])

    for fn in files:
        path = fn if os.path.isabs(fn) else os.path.join(noisy_dir, fn)
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        X_cnn, y_enc, _ = load_and_prepare(path, feature_cols, scaler, label_enc)
        y_pred = np.argmax(model.predict(X_cnn), axis=1)
        acc = accuracy_score(y_enc, y_pred)
        bal = balanced_accuracy_score(y_enc, y_pred)
        f1 = f1_score(y_enc, y_pred, average='macro')
        results.append({'label': os.path.splitext(os.path.basename(path))[0], 'file': path, 'accuracy': acc, 'balanced_accuracy': bal, 'f1_macro': f1})
        print(f"Evaluated {path}: acc={acc:.4f}, bal_acc={bal:.4f}, f1_macro={f1:.4f}")

    # Create DataFrame of results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, 'noisy_evaluation_results.csv'), index=False)
    print(f"Saved results CSV -> {os.path.join(out_dir, 'noisy_evaluation_results.csv')}")

    # Plot accuracies
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res_df, x='label', y='accuracy', palette='viridis')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy: Clean vs Noisy Client_C')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'accuracy_clean_vs_noisy.png')
    plt.savefig(plot_path)
    print(f"Saved accuracy plot -> {plot_path}")

    # Also plot drop from clean
    res_df['drop_from_clean'] = res_df['accuracy'].iloc[0] - res_df['accuracy']
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res_df, x='label', y='drop_from_clean', palette='magma')
    plt.ylabel('Accuracy Drop (from clean)')
    plt.title('Accuracy Drop on Noisy Datasets')
    plt.xticks(rotation=30)
    plt.tight_layout()
    drop_plot_path = os.path.join(out_dir, 'accuracy_drop.png')
    plt.savefig(drop_plot_path)
    print(f"Saved accuracy drop plot -> {drop_plot_path}")

    return res_df


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate saved CNN model on noisy client_C datasets and plot accuracy differences')
    p.add_argument('--model-path', '-m', type=str, default='wisdm_cnn_fft_model.keras', help='Path to saved Keras model')
    p.add_argument('--original', '-o', type=str, default='../WISDM_ar_v1.1/client_C.csv', help='Path to original client_C.csv')
    p.add_argument('--noisy-dir', '-n', type=str, default='noisy_sets', help='Directory containing noisy CSVs')
    p.add_argument('--noisy-files', '-f', type=str, nargs='*', help='Specific noisy files to evaluate (optional)')
    p.add_argument('--out-dir', '-d', type=str, default='eval_results', help='Directory to save results and plots')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    df_results = evaluate_model(model_path=args.model_path, original_csv=args.original, noisy_dir=args.noisy_dir, noisy_files=args.noisy_files, out_dir=args.out_dir)
    print('\nSummary:')
    print(df_results[['label','accuracy','balanced_accuracy','f1_macro']])
