# Telecom Network Incident & Fault Severity Analysis

A production-style data analytics and machine learning project that analyzes telecom network incidents, identifies operational patterns associated with severe faults, and predicts incident fault severity using Python, SQL, SQLite, and machine learning.

The project uses the **Telstra Network Disruptions dataset** and demonstrates an end-to-end workflow from raw relational data through SQL analysis, feature engineering, predictive modelling, model interpretation, and visualization.

---

## Project Objectives

Telecommunications networks generate large volumes of alarms, events, logs, and resource information during operational incidents.

This project investigates three main questions:

1. Which network locations experience the highest incident volumes and highest rates of severe faults?
2. Which event, resource, and log characteristics are associated with fault severity?
3. Can network incident data be used to predict fault severity accurately enough to support operational prioritization?

The project combines traditional operational analytics with machine learning to explore these questions.

---

## Technology Stack

- **Python**
- **Pandas**
- **SQL**
- **SQLite**
- **Scikit-learn**
- **Matplotlib**
- **Git / GitHub**
- **VS Code**

Machine learning techniques include:

- Logistic Regression
- Random Forest
- Class weighting for imbalanced classification
- One-hot encoding
- Feature scaling
- Feature importance analysis
- Multiclass model evaluation

---

## Dataset

The project uses the Telstra Network Disruptions dataset.

The raw data consists of multiple related tables describing network incidents:

- `train.csv`
- `event_type.csv`
- `log_feature.csv`
- `resource_type.csv`
- `severity_type.csv`

The training dataset contains:

- **7,381 labelled network incidents**
- **929 locations**
- Three fault-severity classes

### Target Distribution

| Fault Severity | Incidents | Percentage |
|---|---:|---:|
| Class 0 | 4,784 | 64.82% |
| Class 1 | 1,871 | 25.35% |
| Class 2 | 726 | 9.84% |

The target is therefore imbalanced, making accuracy alone insufficient for evaluating model performance.

![Fault Severity Distribution](reports/figures/fault_severity_distribution.png)

---

## Data Architecture

The original dataset is relational.

A single incident may be associated with multiple:

- event types
- log features
- resource types

Conceptually:

```text
                        INCIDENT
                           |
              +------------+------------+
              |            |            |
              v            v            v
          EVENT TYPE   LOG FEATURE   RESOURCE TYPE
                           |
                           v
                     LOG VOLUME
```

The project therefore performs both relational SQL analysis and incident-level feature engineering before machine learning.

---

## Project Workflow

```text
Raw Telecom Data
        |
        v
Data Exploration & Validation
        |
        v
SQLite Database
        |
        v
SQL Operational Analysis
        |
        v
Incident-Level Aggregation
        |
        v
Feature Engineering
        |
        v
459 Modelling Features
        |
        v
Baseline Logistic Regression
        |
        v
Enhanced Logistic Regression
        |
        v
Random Forest
        |
        v
Model Evaluation
        |
        v
Feature Importance
        |
        v
Operational Insights & Visualizations
```

---

## SQL Analysis

The raw CSV datasets are loaded into a SQLite database to support relational analysis.

SQL techniques used include:

- `JOIN`
- `GROUP BY`
- `CASE WHEN`
- aggregate functions
- Common Table Expressions (CTEs)
- conditional aggregation
- incident-level feature construction

The analysis identifies network locations with unusually high incident volumes and concentrations of severe faults.

### Highest Incident-Volume Locations

![Top Incident Locations](reports/figures/top_incident_locations.png)

The highest-volume location contained **85 incidents**.

High incident frequency, however, does not necessarily imply the highest severe-fault risk.

---

## Severe Fault Concentration by Location

Locations were also evaluated according to their proportion of Severity Class 2 incidents.

To reduce instability from very small samples, the analysis used locations with at least 20 incidents for the severity-rate comparison.

A notable example was:

```text
Location: location 1100
Total incidents: 45
Severity Class 2 incidents: 33
Severity Class 2 rate: 73.33%
```

This demonstrates why incident frequency and incident severity should be evaluated separately.

![Severity Class 2 Rate by Location](reports/figures/severity_2_rate_by_location.png)

---

## Log Activity and Fault Severity

Incident-level log statistics were generated using SQL and Python.

Average log activity differed substantially across severity classes:

| Fault Severity | Avg. Log Features | Avg. Total Log Volume | Avg. Log Volume | Avg. Max Log Volume |
|---|---:|---:|---:|---:|
| Class 0 | 3.26 | 35.94 | 9.88 | 15.63 |
| Class 1 | 3.03 | 13.36 | 4.13 | 7.21 |
| Class 2 | 3.58 | **52.26** | **14.17** | **29.42** |

Severity Class 2 incidents displayed the highest average total, average, and maximum log volumes in this dataset.

![Log Volume by Severity](reports/figures/log_volume_by_severity.png)

These patterns are descriptive associations and should not be interpreted as evidence of causality.

---

## Feature Engineering

The first incident-level dataset contained aggregate operational features such as:

- log feature count
- total log volume
- average log volume
- maximum log volume
- event count
- unique event types
- resource count
- unique resource types
- location
- severity type

This produced a compact incident-level representation.

A second feature-engineering stage retained the identities of individual network signals.

For example:

```text
Raw event records

id      event_type
5022    event_type 15
5022    event_type 11
```

were transformed conceptually into:

```text
id      event_11    event_15
5022       1           1
```

Log features retained their associated volumes:

```text
id      log_56      log_172
5022       1            2
```

The final enhanced dataset contained:

- **7,381 incidents**
- **461 total columns**
- **459 modelling features**
- **0 missing values**
- **0 duplicate incident IDs**

---

## Machine Learning

The target variable is a three-class fault-severity classification problem.

Because the classes are imbalanced, models were evaluated using several metrics rather than accuracy alone:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Confusion matrix

Special attention was given to **Class 2 recall and F1-score** because Class 2 is the smallest target class.

---

## Model 1 — Baseline Logistic Regression

The first Logistic Regression model used aggregate incident features.

Results:

| Metric | Score |
|---|---:|
| Accuracy | 0.619 |
| Macro F1 | 0.553 |
| Class 2 Precision | 0.335 |
| Class 2 Recall | 0.752 |
| Class 2 F1 | 0.464 |

The model achieved strong minority-class recall but produced many false Class 2 predictions.

---

## Model 2 — Enhanced Logistic Regression

The enhanced model incorporated the identities of event types, resource types, and individual log features.

Results:

| Metric | Score |
|---|---:|
| Accuracy | **0.680** |
| Macro F1 | **0.626** |
| Class 2 Precision | **0.446** |
| Class 2 Recall | **0.772** |
| Class 2 F1 | **0.566** |

Feature engineering therefore improved both overall performance and minority-class classification.

Macro F1 increased from:

```text
0.553 -> 0.626
```

without changing the underlying classification algorithm.

This illustrates the importance of feature representation in machine learning.

---

## Model 3 — Random Forest

A Random Forest classifier was then trained using the enhanced feature set.

Results:

| Metric | Score |
|---|---:|
| Accuracy | **0.726** |
| Macro F1 | **0.680** |
| Class 2 Precision | **0.538** |
| Class 2 Recall | **0.786** |
| Class 2 F1 | **0.639** |

Random Forest produced the strongest overall performance.

### Confusion Matrix

```text
                 Predicted
               0     1     2

Actual 0      711   186    60
Actual 1       89   248    38
Actual 2        6    25   114
```

Of the **145 actual Class 2 incidents in the test set, 114 were correctly identified**, corresponding to **78.6% recall**.

---

## Model Comparison

| Model | Accuracy | Macro F1 | Class 2 F1 |
|---|---:|---:|---:|
| Logistic Regression — Aggregate | 0.619 | 0.553 | 0.464 |
| Logistic Regression — Enhanced | 0.680 | 0.626 | 0.566 |
| **Random Forest — Enhanced** | **0.726** | **0.680** | **0.639** |

![Model Performance Comparison](reports/figures/model_comparison.png)

The results show two clear improvements:

```text
Better feature representation
        |
        v
Macro F1: 0.553 -> 0.626

Nonlinear ensemble modelling
        |
        v
Macro F1: 0.626 -> 0.680
```

---

## Feature Importance

Random Forest feature importance was used to investigate which variables contributed most strongly to the fitted model's predictions.

Top features included:

| Feature | Importance |
|---|---:|
| `log_203` | 0.0784 |
| Total log volume | 0.0733 |
| Average log volume | 0.0709 |
| Maximum log volume | 0.0591 |
| `log_82` | 0.0438 |
| Log feature count | 0.0328 |
| `event_15` | 0.0199 |
| `log_312` | 0.0164 |
| `event_35` | 0.0149 |
| `log_170` | 0.0129 |

![Feature Importance](reports/figures/feature_importance.png)

The model relied substantially on both **log-feature identity and log-volume characteristics**.

Feature importance represents predictive contribution within the fitted model and should not be interpreted as causal evidence.

---

## Key Findings

The analysis produced several operationally relevant findings:

1. Network incident volume is concentrated in particular locations, but high volume does not necessarily correspond to the highest severe-fault rate.
2. Severity Class 2 incidents exhibited substantially higher average log-volume characteristics than Class 1 incidents.
3. Preserving individual event, resource, and log-feature identities materially improved classification performance.
4. Enhanced Logistic Regression increased Macro F1 from **0.553 to 0.626**.
5. Random Forest further increased Macro F1 to **0.680**.
6. The final Random Forest identified **78.6% of Class 2 incidents** in the held-out test set.
7. Log-feature identities and log-volume characteristics were among the most important predictive variables.

These results suggest that combining network topology/location information with detailed event and log telemetry can provide useful signals for operational fault-severity prioritization.

---

## Repository Structure

```text
telecom-incident-analysis/
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- reports/
|   `-- figures/
|       |-- fault_severity_distribution.png
|       |-- feature_importance.png
|       |-- log_volume_by_severity.png
|       |-- model_comparison.png
|       |-- severity_2_rate_by_location.png
|       `-- top_incident_locations.png
|
|-- sql/
|   `-- 01_incident_analysis.sql
|
|-- src/
|   |-- 01_data_exploration.py
|   |-- 02_create_database.py
|   |-- 03_run_sql_analysis.py
|   |-- 04_feature_analysis.py
|   |-- 05_baseline_model.py
|   |-- 06_enhanced_features.py
|   |-- 07_enhanced_model.py
|   |-- 08_random_forest_model.py
|   |-- 09_feature_importance.py
|   `-- 10_visualizations.py
|
|-- .gitignore
|-- requirements.txt
`-- README.md
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd telecom-incident-analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the pipeline

```bash
python src/01_data_exploration.py
python src/02_create_database.py
python src/03_run_sql_analysis.py
python src/04_feature_analysis.py
python src/05_baseline_model.py
python src/06_enhanced_features.py
python src/07_enhanced_model.py
python src/08_random_forest_model.py
python src/09_feature_importance.py
python src/10_visualizations.py
```

---

## Skills Demonstrated

This project demonstrates practical experience in:

**Python**
- data manipulation
- reusable analytical scripts
- feature engineering
- machine learning pipelines
- model evaluation

**SQL**
- joins
- aggregations
- CTEs
- conditional aggregation
- relational incident analysis

**Machine Learning**
- multiclass classification
- imbalanced datasets
- Logistic Regression
- Random Forest
- feature encoding
- model comparison
- precision/recall/F1 analysis
- feature importance

**Data Engineering**
- multi-table datasets
- SQLite
- raw/processed data separation
- relational-to-feature-matrix transformation

**Operational Analytics**
- incident concentration
- fault-severity analysis
- log-volume analysis
- operational risk prioritization

---

## Potential Future Improvements

Possible extensions include:

- cross-validation and hyperparameter optimization
- gradient-boosted tree models
- probability calibration
- SHAP-based local and global model interpretation
- automated unit and data-quality tests
- experiment tracking
- model serialization and inference API
- Docker packaging
- CI/CD pipeline
- monitoring for model and data drift

---

## Disclaimer

This project is an independent portfolio analysis using a public telecom network-disruption dataset.

The findings represent patterns observed within this dataset and should not be interpreted as causal relationships or production network recommendations without further validation.

---

## Author

**Shadreck Chabaya**

Data Science | Machine Learning | Telecom Operations & Reliability Analytics