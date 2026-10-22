
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. Load your existing dataset
df = pd.read_excel("../data/raw/final_dataset.xlsx")

features = ["Injection_Rate", "Injection_Pressure"]
X = df[features].apply(pd.to_numeric, errors="coerce")

# Keep valid sensor readings
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()

# 2. Train Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X)
scores = model.decision_function(X)

# 3. Store predictions and scores
results = X.copy()
results["Prediction"] = predictions
results["Anomaly_Score"] = scores

results["Status"] = results["Prediction"].map({
    1: "Normal",
    -1: "Anomaly"
})

# 4. Compare score distributions
print("ANOMALY SCORE SUMMARY")
print(
    results.groupby("Status")["Anomaly_Score"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(6)
)

print("\nFirst 10 detected anomalies:")
print(
    results[results["Prediction"] == -1]
    .sort_values("Anomaly_Score")
    .head(10)
    .to_string()
)

# 5. Plot score distributions
plt.figure(figsize=(9, 5))

for status in ["Normal", "Anomaly"]:
    values = results.loc[
        results["Status"] == status, "Anomaly_Score"
    ]
    plt.hist(values, bins=30, alpha=0.6, label=status)

plt.axvline(0, linestyle="--", label="Decision boundary")
plt.xlabel("Isolation Forest decision score")
plt.ylabel("Number of observations")
plt.title("Isolation Forest Anomaly Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("anomaly_score_distribution.png", dpi=150)
plt.show()

# 6. Save results in new files
results.to_csv("anomaly_score_results.csv", index=False)

print("\nCreated: anomaly_score_results.csv")
print("Created: anomaly_score_distribution.png")
