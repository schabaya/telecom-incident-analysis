import pandas as pd


features = pd.read_csv("data/processed/incident_features.csv")

numeric_features = [
    "log_feature_count",
    "total_log_volume",
    "avg_log_volume",
    "max_log_volume",
    "event_count",
    "unique_event_types",
    "resource_count",
    "unique_resource_types",
]

print("\n--- NUMERICAL FEATURE SUMMARY ---")
print(features[numeric_features].describe().round(2))

print("\n--- FEATURE MEANS BY FAULT SEVERITY ---")

severity_summary = (
    features
    .groupby("fault_severity")[numeric_features]
    .mean()
    .round(2)
)

print(severity_summary)

print("\n--- LOCATION CARDINALITY ---")
print(features["location"].nunique())

print("\n--- SEVERITY TYPE DISTRIBUTION ---")
print(features["severity_type"].value_counts())

print("\n--- TARGET DISTRIBUTION (%) ---")
print(
    features["fault_severity"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)