
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_excel("../data/raw/final_dataset.xlsx")

features = ["Injection_Rate", "Injection_Pressure"]
X = df[features].apply(pd.to_numeric, errors="coerce")

# Keep valid sensor readings
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()

print("Rows used:", len(X))

# 2. Scale features for LOF and One-Class SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Run the three algorithms
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

results = X.copy()
predictions = {}

for name, model in models.items():
    if name == "Isolation Forest":
        pred = model.fit_predict(X)
    else:
        pred = model.fit_predict(X_scaled)

    predictions[name] = pred
    results[name + "_Prediction"] = pred

    print(f"{name}: {(pred == -1).sum()} anomalies")

# 4. Plot each algorithm's detections
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, pred) in zip(axes, predictions.items()):
    normal = pred == 1
    anomaly = pred == -1

    ax.scatter(
        X.loc[normal, "Injection_Rate"],
        X.loc[normal, "Injection_Pressure"],
        alpha=0.45,
        s=18,
        label="Normal"
    )

    ax.scatter(
        X.loc[anomaly, "Injection_Rate"],
        X.loc[anomaly, "Injection_Pressure"],
        alpha=0.85,
        s=25,
        label="Anomaly"
    )

    ax.set_title(name)
    ax.set_xlabel("Injection Rate")
    ax.set_ylabel("Injection Pressure")
    ax.legend()

fig.suptitle("Anomaly Pattern Comparison")
fig.tight_layout()
fig.savefig("algorithm_pattern_comparison.png", dpi=150)
plt.show()

# 5. Save predictions separately
results.to_csv("algorithm_pattern_comparison.csv", index=False)

print("\nCreated: algorithm_pattern_comparison.png")
print("Created: algorithm_pattern_comparison.csv")
