# %% 0. Import libraries
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% 1. Globals
WISDM_DATASET = "WISDM_ar_v1.1"
WISDM_DATASET_RAW = f"{WISDM_DATASET}/{WISDM_DATASET}_raw.txt"
WISDM_DATASET_TRANSFORMED = f"{WISDM_DATASET}/{WISDM_DATASET}_transformed.arff"
PLOT_DIR = "plots"

# Create plots directory if it doesn't exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created directory: {PLOT_DIR}")


# %% 2. Load Data Functions
def load_raw_data(file_path="./WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"):
    data = []
    with open(file_path, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            # line format: [user-id],[activity],[timestamp],[x-acceleration],[y-acceleration],[z-acceleration];
            parts = line.strip().replace(";", "").split(",")
            if len(parts) < 6:  # incomplete line
                continue

            cleaned_parts = []
            for part in parts[:6]:
                part = part.strip()
                if not part:
                    cleaned_parts.append(np.nan)
                else:
                    cleaned_parts.append(part)

            data.append(cleaned_parts)

    columns = ["user_id", "activity", "timestamp", "x_accel", "y_accel", "z_accel"]
    df = pd.DataFrame(data, columns=columns)

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")

    df = df.drop("timestamp", axis=1)

    for col in ["x_accel", "y_accel", "z_accel"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    df["user_id"] = df["user_id"].astype(int)

    return df


def load_transformed_data(file_path="./WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff"):
    with open(file_path, "r") as f:
        content = f.read()

    # Split into header and data sections
    parts = content.split("@data")
    header = parts[0]
    data_section = parts[1].strip()

    # Extract column names from @attribute lines
    attr_pattern = r'@attribute\s+(?:"([^"]+)"|([^\s{]+))'
    columns = [
        match[0] if match[0] else match[1] for match in re.findall(attr_pattern, header)
    ]

    print(f"Found {len(columns)} columns: {columns}")

    # Parse data rows
    rows = []
    for line in data_section.split("\n"):
        line = line.strip()
        if line and not line.startswith("%"):
            # Split by comma, but be careful with quoted values
            values = []
            current = ""
            in_quotes = False

            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == "," and not in_quotes:
                    values.append(current.strip())
                    current = ""
                    continue
                current += char

            if current:
                values.append(current.strip())

            if len(values) == len(columns):
                rows.append(values)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Convert numeric columns
    for col in df.columns:
        if col not in ["user", "class"]:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except:
                pass

    # Drop rows with missing values
    # return df.dropna()

    # Fill missing values with mean (per numeric column)
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=["float64", "int64"]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
        df_clean[numeric_cols].mean()
    )
    return df_clean


# %% 3. Execute Loading
df_raw = load_raw_data(WISDM_DATASET_RAW)
df_trans = load_transformed_data(WISDM_DATASET_TRANSFORMED)

# %% 4. EDA Check 1: Subject Distribution (Using RAW Data)
if df_raw is not None:
    plt.figure(figsize=(14, 6))
    sns.countplot(x="user_id", data=df_raw)
    plt.title("Raw Data Samples per User (Federated Client Simulation)")
    plt.xlabel("User ID")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)


    save_path = os.path.join(PLOT_DIR, "1_user_distribution.png")
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()


    print("\nUser Data Counts (Top 5 & Bottom 5):")
    counts = df_raw["user_id"].value_counts()
    print(pd.concat([counts.head(), counts.tail()]))

# %% 5. EDA Check 2: Class Balance (Using TRANSFORMED Data)
if df_trans is not None:
    plt.figure(figsize=(10, 6))
    # Order bars by count
    sns.countplot(
        y="class",
        data=df_trans,
        order=df_trans["class"].value_counts().index,
    )
    plt.title("Activity Class Distribution (Fairness Check)")
    plt.xlabel("Count")

    save_path = os.path.join(PLOT_DIR, "2_class_balance.png")
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

# %% 6. EDA Check 3: Feature Correlation (Using TRANSFORMED Data)
if df_trans is not None:
    df_numeric = df_trans.drop(["class", "UNIQUE_ID", "user"], axis=1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_numeric.iloc[:, :].corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Matrix (Features)")

    save_path = os.path.join(PLOT_DIR, "3_feature_correlation.png")
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")
    plt.close()

# %% 7. Stacked Bar Graph: Users vs Activities
plot_df = df_trans

# Rows = Users, Columns = Activities, Values = Count
user_activity_counts = pd.crosstab(plot_df["user"], plot_df["class"])

print("\n" + "=" * 50)
print("DEBUG TABLE: Activity Counts per User (Transformed Data)")
print("=" * 50)
print(user_activity_counts)
print("=" * 50 + "\n")

plt.figure(figsize=(16, 8))
user_activity_counts.plot(
    kind="bar", stacked=True, figsize=(16, 8), colormap="tab10", width=0.8
)

plt.title("Activity Distribution per User (Transformed Data)", fontsize=16)
plt.xlabel("User ID", fontsize=14)
plt.ylabel("Number of Samples", fontsize=14)
plt.legend(title="Activity", bbox_to_anchor=(1.0, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()

save_path = os.path.join(PLOT_DIR, "4_user_activity_stacked.png")
plt.savefig(save_path)
print(f"Saved plot: {save_path}")
plt.close()

# %% 8. Verify Federated Client Splits
client_a_users = ["1", "2", "3", "4", "9", "12", "17", "19", "20", "25", "26", "31"]
client_b_users = ["5", "10", "11", "13", "14", "16", "21", "27", "29", "28", "30"]
client_c_users = ["6", "7", "8", "15", "18", "22", "23", "24", "32", "33", "34", "35", "36"]

user_to_client = {}
for u in client_a_users:
    user_to_client[u] = "Client A"
for u in client_b_users:
    user_to_client[u] = "Client B"
for u in client_c_users:
    user_to_client[u] = "Client C"

client_df = plot_df.copy()

client_df["user"] = client_df["user"].map(user_to_client).fillna("Unknown")

client_activity_counts = pd.crosstab(client_df["user"], client_df["class"])

print("\n" + "=" * 50)
print("FEDERATED CLIENT DISTRIBUTION TABLE")
print("=" * 50)
print(client_activity_counts)
print("=" * 50 + "\n")

plt.figure(figsize=(10, 8))
client_activity_counts.plot(
    kind="bar", stacked=True, figsize=(10, 8), colormap="tab10", width=0.5
)

plt.title("Activity Distribution per Federated Client", fontsize=16)
plt.xlabel("Federated Client", fontsize=14)
plt.ylabel("Number of Samples", fontsize=14)
plt.xticks(rotation=0)
plt.legend(title="Activity", bbox_to_anchor=(1.0, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()

save_path = os.path.join(PLOT_DIR, "5_client_distribution_check.png")
plt.savefig(save_path)
print(f"Saved Client Verification Plot: {save_path}")
plt.close()

# %% 9. Save Federated Client Datasets (Final Step)
OUTPUT_DIR = WISDM_DATASET

df_to_split = plot_df

splits = {
    "client A": client_a_users,
    "client B": client_b_users,
    "client C": client_c_users
}

print(f"\nSaving files to '{OUTPUT_DIR}/'...")

for client_name, users in splits.items():
    client_subset = df_to_split[df_to_split['user'].isin(users)].copy()

    filename = f"{client_name.replace(' ', '_')}.csv"
    save_path = os.path.join(OUTPUT_DIR, filename)

    client_subset.to_csv(save_path, index=False)

    print(f" -> Saved {filename}")
    print(f"    Rows: {client_subset.shape[0]}, Columns: {client_subset.shape[1]}")
    print("-" * 30)

print("\nEDA Complete. Check the 'plots/' folder for results.")
