"""
create_noisy_client_c.py

Create noisy variants of the WISDM `client_C.csv` dataset for robustness testing.

Usage examples:
    python create_noisy_client_c.py --input ../WISDM_ar_v1.1/client_C.csv --noise-levels 0.05 0.1 0.2
    python create_noisy_client_c.py --input ../WISDM_ar_v1.1/client_C.csv --noise-levels 0.05 0.1 --output-dir noisy_sets --seed 42

The script saves CSV files named like: client_C_noise_0.050.csv in the output directory.
"""

import os
import argparse
from typing import List, Optional

import pandas as pd
import numpy as np


def add_noise(df: pd.DataFrame, noise_level: float = 0.1, non_feature_cols: Optional[List[str]] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """Adds Gaussian noise to numeric feature columns of the dataframe.

    Parameters
    - df: input DataFrame
    - noise_level: standard deviation of Gaussian noise (absolute scale)
    - non_feature_cols: list of columns to exclude from noise (IDs/labels)
    - seed: optional random seed for reproducibility

    Returns: noisy copy of df
    """
    if seed is not None:
        np.random.seed(seed)

    noisy_df = df.copy()

    if non_feature_cols is None:
        non_feature_cols = ['UNIQUE_ID', 'user', 'class']

    # Select feature columns to add noise to (numeric only)
    feature_cols = [c for c in noisy_df.columns if c not in non_feature_cols]
    # Keep only numeric columns among features
    numeric_cols = noisy_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError('No numeric feature columns found to add noise to.')

    noise = np.random.normal(loc=0.0, scale=noise_level, size=noisy_df[numeric_cols].shape)
    noisy_df[numeric_cols] = noisy_df[numeric_cols].values + noise

    return noisy_df


def generate_and_save(input_path: str, noise_levels: List[float], output_dir: str = 'noisy_versions', seed: Optional[int] = None, non_feature_cols: Optional[List[str]] = None):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for i, nl in enumerate(noise_levels):
        s = None if seed is None else int(seed + i)
        noisy = add_noise(df, noise_level=float(nl), non_feature_cols=non_feature_cols, seed=s)
        out_name = f"{base_name}_noise_{float(nl):.3f}.csv"
        out_path = os.path.join(output_dir, out_name)
        noisy.to_csv(out_path, index=False)
        print(f"Saved noisy dataset (noise={nl}) -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Create noisy variants of client_C.csv for robustness testing.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to client_C.csv')
    parser.add_argument('--noise-levels', '-n', type=float, nargs='+', required=True, help='One or more noise stddev values, e.g. 0.05 0.1 0.2')
    parser.add_argument('--output-dir', '-o', type=str, default='noisy_versions', help='Directory to save noisy CSVs')
    parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--exclude-cols', '-e', type=str, nargs='*', default=['UNIQUE_ID', 'user', 'class'], help='Columns to exclude from adding noise')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_and_save(input_path=args.input, noise_levels=args.noise_levels, output_dir=args.output_dir, seed=args.seed, non_feature_cols=args.exclude_cols)
