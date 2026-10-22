
import pandas as pd

# Load the comparison results already created
df = pd.read_csv("algorithm_comparison.csv")

# Prediction columns
if_col = "Isolation Forest_Prediction"
lof_col = "LOF_Prediction"
svm_col = "One-Class SVM_Prediction"

models = [if_col, lof_col, svm_col]

# Count observations flagged by each algorithm
print("ANOMALY COUNTS")
for col in models:
    print(f"{col}: {(df[col] == -1).sum()}")

# Count observations where all algorithms agree
all_normal = (df[models] == 1).all(axis=1).sum()
all_anomaly = (df[models] == -1).all(axis=1).sum()

print("\nAGREEMENT RESULTS")
print("All three say normal:", all_normal)
print("All three say anomaly:", all_anomaly)

# Compare every pair of algorithms
print("\nPAIRWISE AGREEMENT")
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        agreement = (df[models[i]] == df[models[j]]).mean() * 100
        print(
            f"{models[i]} vs {models[j]}: "
            f"{agreement:.2f}% agreement"
        )

# Save a copy of observations flagged by at least one algorithm
flagged = df[(df[models] == -1).any(axis=1)]
flagged.to_csv("flagged_observations.csv", index=False)

print("\nCreated: flagged_observations.csv")
