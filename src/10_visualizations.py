from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# 1. FILE PATHS
# ---------------------------------------------------------

DATA_PATH = Path("data/processed")
FIGURE_PATH = Path("reports/figures")

# Create the figures directory if it does not already exist.
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

features = pd.read_csv(
    DATA_PATH / "incident_features.csv"
)

feature_importance = pd.read_csv(
    DATA_PATH / "feature_importance.csv"
)


# ---------------------------------------------------------
# 3. FAULT SEVERITY DISTRIBUTION
# ---------------------------------------------------------

severity_counts = (
    features["fault_severity"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

severity_counts.plot(
    kind="bar"
)

plt.title("Fault Severity Distribution")
plt.xlabel("Fault Severity Class")
plt.ylabel("Number of Incidents")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "fault_severity_distribution.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 4. TOP 10 LOCATIONS BY INCIDENT COUNT
# ---------------------------------------------------------

top_locations = (
    features["location"]
    .value_counts()
    .head(10)
    .sort_values()
)

plt.figure(figsize=(9, 6))

top_locations.plot(
    kind="barh"
)

plt.title("Top 10 Locations by Incident Count")
plt.xlabel("Number of Incidents")
plt.ylabel("Location")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "top_incident_locations.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 5. LOCATIONS WITH HIGHEST CLASS 2 RATE
# ---------------------------------------------------------

location_summary = (
    features
    .groupby("location")
    .agg(
        total_incidents=(
            "fault_severity",
            "size",
        ),
        severity_2=(
            "fault_severity",
            lambda x: (x == 2).sum(),
        ),
    )
    .reset_index()
)

location_summary["severity_2_pct"] = (
    100
    * location_summary["severity_2"]
    / location_summary["total_incidents"]
)

# Only compare locations with at least 20 incidents.
location_summary = location_summary[
    location_summary["total_incidents"] >= 20
]

top_severity_locations = (
    location_summary
    .sort_values(
        "severity_2_pct",
        ascending=False,
    )
    .head(10)
    .sort_values("severity_2_pct")
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_severity_locations["location"],
    top_severity_locations["severity_2_pct"],
)

plt.title(
    "Locations with Highest Severity Class 2 Rate"
)
plt.xlabel("Severity Class 2 Incidents (%)")
plt.ylabel("Location")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "severity_2_rate_by_location.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 6. AVERAGE TOTAL LOG VOLUME BY FAULT SEVERITY
# ---------------------------------------------------------

log_volume_by_severity = (
    features
    .groupby("fault_severity")[
        "total_log_volume"
    ]
    .mean()
)

plt.figure(figsize=(8, 5))

log_volume_by_severity.plot(
    kind="bar"
)

plt.title(
    "Average Total Log Volume by Fault Severity"
)
plt.xlabel("Fault Severity Class")
plt.ylabel("Average Total Log Volume")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "log_volume_by_severity.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 7. TOP 15 RANDOM FOREST FEATURE IMPORTANCES
# ---------------------------------------------------------

top_features = (
    feature_importance
    .head(15)
    .copy()
)

# Remove sklearn transformer prefixes for readability.
top_features["feature"] = (
    top_features["feature"]
    .str.replace(
        "num__",
        "",
        regex=False,
    )
    .str.replace(
        "cat__",
        "",
        regex=False,
    )
)

top_features = top_features.sort_values(
    "importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"],
)

plt.title(
    "Top 15 Random Forest Feature Importances"
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "feature_importance.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 8. MODEL COMPARISON
# ---------------------------------------------------------

model_results = pd.DataFrame(
    {
        "Model": [
            "Logistic - Aggregate",
            "Logistic - Enhanced",
            "Random Forest",
        ],
        "Accuracy": [
            0.619,
            0.680,
            0.726,
        ],
        "Macro F1": [
            0.553,
            0.626,
            0.680,
        ],
    }
)

model_results = model_results.set_index(
    "Model"
)

plt.figure(figsize=(10, 6))

model_results.plot(
    kind="bar"
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(
    rotation=15,
    ha="right",
)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "model_comparison.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 9. SAVE ANALYTICAL SUMMARY
# ---------------------------------------------------------

location_summary.to_csv(
    DATA_PATH / "location_severity_summary.csv",
    index=False,
)

model_results.to_csv(
    DATA_PATH / "model_comparison.csv"
)


# ---------------------------------------------------------
# 10. COMPLETION MESSAGE
# ---------------------------------------------------------

print(
    "\nVisualizations created successfully."
)

print(
    f"Figures saved to: {FIGURE_PATH}"
)

print(
    "\nCreated:"
)

for figure in sorted(
    FIGURE_PATH.glob("*.png")
):
    print(f" - {figure.name}")