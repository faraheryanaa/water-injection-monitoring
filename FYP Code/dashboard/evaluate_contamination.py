
import pandas as pd
from sklearn.ensemble import IsolationForest
from itertools import combinations

# 1. Load the dataset
df = pd.read_excel("../data/raw/final_dataset.xlsx")

features = ["Injection_Rate", "Injection_Pressure"]
X = df[features].apply(pd.to_numeric, errors="coerce")

# Use rows with valid sensor readings
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()

# 2. Test different contamination settings
contamination_values = [0.01, 0.03, 0.05, 0.07, 0.10]
anomaly_sets = {}

print("ANOMALY COUNTS")

for contamination in contamination_values:
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(X)
    anomaly_indices = set(X.index[predictions == -1])
    anomaly_sets[contamination] = anomaly_indices

    print(
        f"{contamination:.0%} contamination: "
        f"{len(anomaly_indices)} anomalies"
    )

# 3. Compare overlap between settings
print("\nPAIRWISE ANOMALY OVERLAP")

results = []

for a, b in combinations(contamination_values, 2):
    set_a = anomaly_sets[a]
    set_b = anomaly_sets[b]

    common = len(set_a & set_b)
    union = len(set_a | set_b)

    jaccard = common / union if union else 1.0

    results.append({
        "Contamination A": a,
        "Contamination B": b,
        "Common Anomalies": common,
        "Jaccard Similarity (%)": round(jaccard * 100, 2)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 4. Save results in a new file
results_df.to_csv(
    "contamination_stability_results.csv",
    index=False
)

print("\nCreated: contamination_stability_results.csv")
