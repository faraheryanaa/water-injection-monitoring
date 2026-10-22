
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Load the dataset
df = pd.read_excel("../data/raw/final_dataset.xlsx")

features = ["Injection_Rate", "Injection_Pressure"]
X = df[features].apply(pd.to_numeric, errors="coerce")

# Keep valid sensor readings
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()

# 2. Run Isolation Forest using the current settings
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X)

# 3. Separate normal and detected anomaly observations
normal = X.loc[predictions == 1]
anomalies = X.loc[predictions == -1]

print("SENSOR PATTERN SUMMARY")
print("Normal observations:", len(normal))
print("Detected anomalies:", len(anomalies))

print("\nDetected anomaly readings:")
print(
    anomalies.describe().round(3).to_string()
)

# 4. Plot injection rate against pressure
plt.figure(figsize=(9, 6))

plt.scatter(
    normal["Injection_Rate"],
    normal["Injection_Pressure"],
    alpha=0.5,
    label="Normal"
)

plt.scatter(
    anomalies["Injection_Rate"],
    anomalies["Injection_Pressure"],
    alpha=0.8,
    label="Detected anomaly"
)

plt.xlabel("Injection Rate")
plt.ylabel("Injection Pressure")
plt.title("Injection Rate vs Pressure: Isolation Forest")
plt.legend()
plt.tight_layout()
plt.savefig("sensor_anomaly_patterns.png", dpi=150)
plt.show()

# 5. Plot each sensor feature separately
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, feature in zip(axes, features):
    ax.boxplot(
        [
            normal[feature],
            anomalies[feature]
        ],
        tick_labels=["Normal", "Anomaly"]
    )
    ax.set_title(feature)
    ax.set_ylabel("Value")

fig.suptitle("Normal vs Detected Anomaly Readings")
fig.tight_layout()
fig.savefig("sensor_anomaly_boxplots.png", dpi=150)
plt.show()

print("\nCreated: sensor_anomaly_patterns.png")
print("Created: sensor_anomaly_boxplots.png")
