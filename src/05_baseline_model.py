import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. LOAD ENGINEERED DATASET
# ---------------------------------------------------------

features = pd.read_csv("data/processed/incident_features.csv")


# ---------------------------------------------------------
# 2. DEFINE FEATURES (X) AND TARGET (y)
# ---------------------------------------------------------

X = features.drop(columns=["id", "fault_severity"])
y = features["fault_severity"]


# ---------------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TEST SETS
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# 4. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ---------------------------------------------------------

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

categorical_features = [
    "location",
    "severity_type",
]


# ---------------------------------------------------------
# 5. CREATE PREPROCESSING PIPELINE
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# 6. CREATE LOGISTIC REGRESSION MODEL PIPELINE
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 7. TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ---------------------------------------------------------
# 8. MAKE PREDICTIONS
# ---------------------------------------------------------

predictions = model.predict(X_test)


# ---------------------------------------------------------
# 9. DISPLAY TEST SET INFORMATION
# ---------------------------------------------------------

print("\n--- TEST SET SIZE ---")
print(len(y_test))

print("\n--- ACTUAL CLASS DISTRIBUTION ---")
print(y_test.value_counts().sort_index())


# ---------------------------------------------------------
# 10. MODEL EVALUATION
# ---------------------------------------------------------

print("\n--- CLASSIFICATION REPORT ---")

print(
    classification_report(
        y_test,
        predictions,
        digits=3,
    )
)


# ---------------------------------------------------------
# 11. CONFUSION MATRIX
# ---------------------------------------------------------

print("\n--- CONFUSION MATRIX ---")

print(
    confusion_matrix(
        y_test,
        predictions,
    )
)