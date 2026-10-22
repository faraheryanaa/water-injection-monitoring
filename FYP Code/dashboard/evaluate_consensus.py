
import pandas as pd

df = pd.read_csv("algorithm_comparison.csv")

prediction_cols = [
    "Isolation Forest_Prediction",
    "LOF_Prediction",
    "One-Class SVM_Prediction"
]

# Count how many algorithms flag each observation
df["Anomaly_Votes"] = (df[prediction_cols] == -1).sum(axis=1)

# Group observations by the number of anomaly votes
print("ANOMALY CONSENSUS")
print("All three normal:",
      (df["Anomaly_Votes"] == 0).sum())
print("Only one algorithm flags anomaly:",
      (df["Anomaly_Votes"] == 1).sum())
print("Two algorithms flag anomaly:",
      (df["Anomaly_Votes"] == 2).sum())
print("All three flag anomaly:",
      (df["Anomaly_Votes"] == 3).sum())

# Save a new file for further inspection
df.to_csv("algorithm_consensus.csv", index=False)

print("\nCreated: algorithm_consensus.csv")
