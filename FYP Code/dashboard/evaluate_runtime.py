
import pandas as pd
import time
import statistics

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_excel("../data/raw/final_dataset.xlsx")

features = ["Injection_Rate", "Injection_Pressure"]
X = df[features].apply(pd.to_numeric, errors="coerce")

# Keep valid rows only
X = X.dropna()

if X.empty:
    raise ValueError("No valid sensor rows found.")

# 2. Prepare inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Define the algorithms
def make_models():
    return {
        "Isolation Forest": (
            IsolationForest(
                contamination=0.05,
                random_state=42
            ),
            X
        ),
        "LOF": (
            LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.05
            ),
            X_scaled
        ),
        "One-Class SVM": (
            OneClassSVM(
                kernel="rbf",
                nu=0.05
            ),
            X_scaled
        )
    }

# 4. Repeat each measurement
repeats = 10
timings = {name: [] for name in make_models()}

for run in range(repeats):
    for name, (model, data) in make_models().items():
        start = time.perf_counter()
        model.fit_predict(data)
        elapsed = time.perf_counter() - start
        timings[name].append(elapsed)

# 5. Summarise runtime
summary = []

for name, times in timings.items():
    summary.append({
        "Algorithm": name,
        "Runs": len(times),
        "Average Runtime (s)": round(statistics.mean(times), 6),
        "Median Runtime (s)": round(statistics.median(times), 6),
        "Minimum Runtime (s)": round(min(times), 6),
        "Maximum Runtime (s)": round(max(times), 6)
    })

summary_df = pd.DataFrame(summary)

print("RUNTIME COMPARISON")
print(summary_df.to_string(index=False))

summary_df.to_csv("runtime_evaluation.csv", index=False)

print("\nCreated: runtime_evaluation.csv")
