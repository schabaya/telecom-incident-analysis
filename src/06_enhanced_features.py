import pandas as pd


# ---------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------

train = pd.read_csv("data/raw/train.csv")
event_type = pd.read_csv("data/raw/event_type.csv")
log_feature = pd.read_csv("data/raw/log_feature.csv")
resource_type = pd.read_csv("data/raw/resource_type.csv")
severity_type = pd.read_csv("data/raw/severity_type.csv")


# ---------------------------------------------------------
# 2. LOAD EXISTING AGGREGATED FEATURES
# ---------------------------------------------------------

base_features = pd.read_csv("data/processed/incident_features.csv")


# ---------------------------------------------------------
# 3. ENCODE EVENT TYPES
# ---------------------------------------------------------

event_encoded = pd.crosstab(
    event_type["id"],
    event_type["event_type"]
)

event_encoded.columns = [
    f"event_{col.replace('event_type ', '')}"
    for col in event_encoded.columns
]

event_encoded = event_encoded.reset_index()


# ---------------------------------------------------------
# 4. ENCODE RESOURCE TYPES
# ---------------------------------------------------------

resource_encoded = pd.crosstab(
    resource_type["id"],
    resource_type["resource_type"]
)

resource_encoded.columns = [
    f"resource_{col.replace('resource_type ', '')}"
    for col in resource_encoded.columns
]

resource_encoded = resource_encoded.reset_index()


# ---------------------------------------------------------
# 5. ENCODE LOG FEATURE IDENTITIES USING VOLUME
# ---------------------------------------------------------

log_encoded = log_feature.pivot_table(
    index="id",
    columns="log_feature",
    values="volume",
    aggfunc="sum",
    fill_value=0,
)

log_encoded.columns = [
    f"log_{col.replace('feature ', '')}"
    for col in log_encoded.columns
]

log_encoded = log_encoded.reset_index()


# ---------------------------------------------------------
# 6. MERGE ENCODED FEATURES
# ---------------------------------------------------------

enhanced_features = base_features.merge(
    event_encoded,
    on="id",
    how="left",
)

enhanced_features = enhanced_features.merge(
    resource_encoded,
    on="id",
    how="left",
)

enhanced_features = enhanced_features.merge(
    log_encoded,
    on="id",
    how="left",
)


# ---------------------------------------------------------
# 7. HANDLE ANY MISSING VALUES
# ---------------------------------------------------------

enhanced_features = enhanced_features.fillna(0)


# ---------------------------------------------------------
# 8. VALIDATE DATASET
# ---------------------------------------------------------

print("\n--- ENHANCED FEATURE DATASET ---")
print(enhanced_features.head())

print("\n--- DATASET SHAPE ---")
print(enhanced_features.shape)

print("\n--- NUMBER OF FEATURES ---")
print(enhanced_features.shape[1] - 2)

print("\n--- MISSING VALUES ---")
print(enhanced_features.isnull().sum().sum())

print("\n--- DUPLICATE INCIDENT IDS ---")
print(enhanced_features["id"].duplicated().sum())


# ---------------------------------------------------------
# 9. SAVE ENHANCED FEATURE DATASET
# ---------------------------------------------------------

enhanced_features.to_csv(
    "data/processed/enhanced_incident_features.csv",
    index=False,
)

print("\nEnhanced feature dataset saved successfully.")