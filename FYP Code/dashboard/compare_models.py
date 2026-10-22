
import pandas as pd
import time

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 1. Load your existing dataset
df = pd.read_excel("../data/raw/final_dataset.xlsx")

# 2. Select the sensor features
features = ["Injection_Rate", "Injection_Pressure"]
X = df[features].copy()

# Check for missing or invalid values
X = X.apply(pd.to_numeric, errors="coerce")
valid_rows = X.notna().all(axis=1)

X_valid = X.loc[valid_rows].copy()

if X_valid.empty:
    raise ValueError("No valid sensor rows were found.")

print("Dataset loaded successfully!")
print("Total rows:", len(df))
print("Rows used for comparison:", len(X_valid))
print("Features:", features)

# 3. Scale features for LOF and One-Class SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)

# 4. Define the three algorithms
models = {
    "Isolation Forest": IsolationForest(
        contamination=0.05,
        random_state=42
    ),
    "LOF": LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.05
    ),
    "One-Class SVM": OneClassSVM(
        kernel="rbf",
        nu=0.05
    )
}

# 5. Compare results
results = df.loc[valid_rows].copy()
summary = []

for name, model in models.items():
    start = time.perf_counter()

    if name == "Isolation Forest":
        predictions = model.fit_predict(X_valid)
    else:
        predictions = model.fit_predict(X_scaled)

    elapsed = time.perf_counter() - start

    # -1 = detected anomaly; 1 = normal
    results[name + "_Prediction"] = predictions

    summary.append({
        "Algorithm": name,
        "Anomalies": int((predictions == -1).sum()),
        "Normal": int((predictions == 1).sum()),
        "Runtime (seconds)": round(elapsed, 6)
    })

summary_df = pd.DataFrame(summary)

print("\nALGORITHM COMPARISON")
print(summary_df.to_string(index=False))

# 6. Save new results without overwriting your existing files
results.to_csv("algorithm_comparison.csv", index=False)
summary_df.to_csv("algorithm_summary.csv", index=False)

print("\nComparison completed!")
print("Created: algorithm_comparison.csv")
print("Created: algorithm_summary.csv")
