import sqlite3

import pandas as pd


connection = sqlite3.connect("data/processed/telecom_incidents.db")

query = """
SELECT
    location,
    COUNT(*) AS incident_count
FROM train
GROUP BY location
ORDER BY incident_count DESC
LIMIT 10;
"""

result = pd.read_sql_query(query, connection)

print("\n--- TOP 10 LOCATIONS BY INCIDENT COUNT ---")
print(result)

query_severity_2 = """
SELECT
    location,
    COUNT(*) AS severity_2_count
FROM train
WHERE fault_severity = 2
GROUP BY location
ORDER BY severity_2_count DESC
LIMIT 10;
"""

severity_2_result = pd.read_sql_query(query_severity_2, connection)

print("\n--- TOP 10 LOCATIONS BY SEVERITY CLASS 2 INCIDENTS ---")
print(severity_2_result)

query_severity_rate = """
SELECT
    location,
    COUNT(*) AS total_incidents,
    SUM(CASE WHEN fault_severity = 0 THEN 1 ELSE 0 END) AS severity_0,
    SUM(CASE WHEN fault_severity = 1 THEN 1 ELSE 0 END) AS severity_1,
    SUM(CASE WHEN fault_severity = 2 THEN 1 ELSE 0 END) AS severity_2,
    ROUND(
        100.0 * SUM(CASE WHEN fault_severity = 2 THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS severity_2_pct
FROM train
GROUP BY location
HAVING COUNT(*) >= 20
ORDER BY severity_2_pct DESC
LIMIT 10;
"""

severity_rate_result = pd.read_sql_query(query_severity_rate, connection)

print("\n--- LOCATIONS WITH HIGHEST SEVERITY CLASS 2 RATE ---")
print(severity_rate_result)

query_log_severity = """
WITH log_summary AS (
    SELECT
        id,
        COUNT(*) AS log_feature_count,
        SUM(volume) AS total_log_volume,
        AVG(volume) AS avg_log_volume,
        MAX(volume) AS max_log_volume
    FROM log_feature
    GROUP BY id
)

SELECT
    t.fault_severity,
    COUNT(*) AS incident_count,
    ROUND(AVG(l.log_feature_count), 2) AS avg_features_per_incident,
    ROUND(AVG(l.total_log_volume), 2) AS avg_total_log_volume,
    ROUND(AVG(l.avg_log_volume), 2) AS avg_log_volume,
    ROUND(AVG(l.max_log_volume), 2) AS avg_max_log_volume
FROM train AS t
JOIN log_summary AS l
    ON t.id = l.id
GROUP BY t.fault_severity
ORDER BY t.fault_severity;
"""

log_severity_result = pd.read_sql_query(query_log_severity, connection)

print("\n--- LOG ACTIVITY BY FAULT SEVERITY ---")
print(log_severity_result)

query = """
WITH log_summary AS (
    SELECT
        id,
        COUNT(*) AS log_feature_count,
        SUM(volume) AS total_log_volume,
        AVG(volume) AS avg_log_volume,
        MAX(volume) AS max_log_volume
    FROM log_feature
    GROUP BY id
),

event_summary AS (
    SELECT
        id,
        COUNT(*) AS event_count,
        COUNT(DISTINCT event_type) AS unique_event_types
    FROM event_type
    GROUP BY id
),

resource_summary AS (
    SELECT
        id,
        COUNT(*) AS resource_count,
        COUNT(DISTINCT resource_type) AS unique_resource_types
    FROM resource_type
    GROUP BY id
)

SELECT
    t.id,
    t.location,
    t.fault_severity,

    l.log_feature_count,
    l.total_log_volume,
    l.avg_log_volume,
    l.max_log_volume,

    e.event_count,
    e.unique_event_types,

    r.resource_count,
    r.unique_resource_types,

    s.severity_type

FROM train AS t

LEFT JOIN log_summary AS l
    ON t.id = l.id

LEFT JOIN event_summary AS e
    ON t.id = e.id

LEFT JOIN resource_summary AS r
    ON t.id = r.id

LEFT JOIN severity_type AS s
    ON t.id = s.id;
"""

features = pd.read_sql_query(query, connection)

connection.close()

print("\n--- ENGINEERED DATASET ---")
print(features.head())

print("\n--- DATASET SHAPE ---")
print(features.shape)

print("\n--- MISSING VALUES ---")
print(features.isnull().sum())

features.to_csv(
    "data/processed/incident_features.csv",
    index=False
)

print("\nFeature dataset saved successfully.")


connection.close()