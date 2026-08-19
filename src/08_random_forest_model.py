import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. LOAD ENHANCED DATASET
# ---------------------------------------------------------

data = pd.read_csv(
    "data/processed/enhanced_incident_features.csv"
)


# ---------------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# ---------------------------------------------------------

X = data.drop(
    columns=[
        "id",
        "fault_severity",
    ]
)

y = data["fault_severity"]


# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# 4. DEFINE CATEGORICAL FEATURES
# ---------------------------------------------------------

categorical_features = [
    "location",
    "severity_type",
]


# ---------------------------------------------------------
# 5. DEFINE NUMERICAL FEATURES
# ---------------------------------------------------------

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


# ---------------------------------------------------------
# 6. PREPROCESSING
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


# ---------------------------------------------------------
# 7. RANDOM FOREST
# ---------------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------
# 8. CREATE PIPELINE
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            classifier,
        ),
    ]
)


# ---------------------------------------------------------
# 9. TRAIN MODEL
# ---------------------------------------------------------

print(
    "\nTraining Random Forest model..."
)

model.fit(
    X_train,
    y_train,
)

print(
    "Model training completed."
)


# ---------------------------------------------------------
# 10. MAKE PREDICTIONS
# ---------------------------------------------------------

predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# 11. CALCULATE METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
)


# ---------------------------------------------------------
# 12. DISPLAY RESULTS
# ---------------------------------------------------------

print(
    "\n--- RANDOM FOREST RESULTS ---"
)

print(
    f"Accuracy: {accuracy:.3f}"
)

print(
    f"Macro F1: {macro_f1:.3f}"
)


# ---------------------------------------------------------
# 13. CLASSIFICATION REPORT
# ---------------------------------------------------------

print(
    "\n--- CLASSIFICATION REPORT ---"
)

print(
    classification_report(
        y_test,
        predictions,
        digits=3,
    )
)


# ---------------------------------------------------------
# 14. CONFUSION MATRIX
# ---------------------------------------------------------

print(
    "\n--- CONFUSION MATRIX ---"
)

print(
    confusion_matrix(
        y_test,
        predictions,
    )
)