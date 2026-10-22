
import pandas as pd
from sklearn.ensemble import IsolationForest
from itertools import combinations

# 1. Load the same dataset
df = pd.read_excel("../data/raw/final_dataset.xlsx")

features = ["Injection_Rate", "Injection_Pressure"]
X = df[features].apply(pd.to_numeric, errors="coerce")

# Use only rows with valid sensor readings
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()

# 2. Run Isolation Forest with different random seeds
seeds = [1, 10, 42, 100, 2026]
predictions = {}

for seed in seeds:
    model = IsolationForest(
        contamination=0.05,
        random_state=seed
    )

    predictions[seed] = model.fit_predict(X)

    print(
        f"Seed {seed}: "
        f"{(predictions[seed] == -1).sum()} anomalies"
    )

# 3. Compare overlap between each pair of runs
print("\nPAIRWISE ANOMALY OVERLAP")

overlap_results = []

for seed_a, seed_b in combinations(seeds, 2):
    anomalies_a = set(X.index[predictions[seed_a] == -1])
    anomalies_b = set(X.index[predictions[seed_b] == -1])

    common = len(anomalies_a & anomalies_b)
    union = len(anomalies_a | anomalies_b)

    # Jaccard similarity: 1 means identical anomaly sets
    similarity = common / union if union else 1.0

    overlap_results.append({
        "Seed A": seed_a,
        "Seed B": seed_b,
        "Common Anomalies": common,
        "Jaccard Similarity (%)": round(similarity * 100, 2)
    })

overlap_df = pd.DataFrame(overlap_results)

print(overlap_df.to_string(index=False))

# 4. Save the new evaluation results
overlap_df.to_csv("stability_results.csv", index=False)

print("\nCreated: stability_results.csv")
