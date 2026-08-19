import sqlite3

import pandas as pd

train = pd.read_csv("data/raw/train.csv")
event_type = pd.read_csv("data/raw/event_type.csv")
log_feature = pd.read_csv("data/raw/log_feature.csv")
resource_type = pd.read_csv("data/raw/resource_type.csv")
severity_type = pd.read_csv("data/raw/severity_type.csv")

connection = sqlite3.connect("data/processed/telecom_incidents.db")

train.to_sql("train", connection, if_exists="replace", index=False)
event_type.to_sql("event_type", connection, if_exists="replace", index=False)
log_feature.to_sql("log_feature", connection, if_exists="replace", index=False)
resource_type.to_sql("resource_type", connection, if_exists="replace", index=False)
severity_type.to_sql("severity_type", connection, if_exists="replace", index=False)

connection.close()

print("Database created successfully.")