import pandas as pd

train = pd.read_csv("data/raw/train.csv")

print(train.head())

print("\n--- DATASET SHAPE ---")
print(train.shape)

print("\n--- COLUMN NAMES ---")
print(train.columns)

print("\n--- DATA TYPES ---")
print(train.dtypes)

print("\n--- MISSING VALUES ---")
print(train.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(train.duplicated().sum())

print("\n--- FAULT SEVERITY DISTRIBUTION ---")
print(train["fault_severity"].value_counts().sort_index())

train["fault_severity"].value_counts()

event_type = pd.read_csv("data/raw/event_type.csv")
log_feature = pd.read_csv("data/raw/log_feature.csv")
resource_type = pd.read_csv("data/raw/resource_type.csv")
severity_type = pd.read_csv("data/raw/severity_type.csv")

print("\n--- EVENT TYPE ---")
print(event_type.shape)
print(event_type.head())

print("\n--- LOG FEATURE ---")
print(log_feature.shape)
print(log_feature.head())

print("\n--- RESOURCE TYPE ---")
print(resource_type.shape)
print(resource_type.head())

print("\n--- SEVERITY TYPE ---")
print(severity_type.shape)
print(severity_type.head())

print("\n--- UNIQUE INCIDENT IDS ---")
print(f"Train: {train['id'].nunique()}")
print(f"Event type: {event_type['id'].nunique()}")
print(f"Log feature: {log_feature['id'].nunique()}")
print(f"Resource type: {resource_type['id'].nunique()}")
print(f"Severity type: {severity_type['id'].nunique()}")

print("\n--- AVERAGE RECORDS PER INCIDENT ---")
print(f"Event types: {len(event_type) / event_type['id'].nunique():.2f}")
print(f"Log features: {len(log_feature) / log_feature['id'].nunique():.2f}")
print(f"Resource types: {len(resource_type) / resource_type['id'].nunique():.2f}")
print(f"Severity types: {len(severity_type) / severity_type['id'].nunique():.2f}")