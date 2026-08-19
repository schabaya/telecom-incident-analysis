import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
# 4. IDENTIFY CATEGORICAL FEATURES
# ---------------------------------------------------------

categorical_features = [
    "location",
    "severity_type",
]


# ---------------------------------------------------------
# 5. IDENTIFY NUMERICAL FEATURES
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
            StandardScaler(),
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
# 7. CREATE MODEL PIPELINE
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
                max_iter=3000,
                class_weight="balanced",
            ),
        ),
    ]
)


# ---------------------------------------------------------
# 8. TRAIN MODEL
# ---------------------------------------------------------

print(
    "\nTraining enhanced Logistic Regression model..."
)

model.fit(
    X_train,
    y_train,
)

print(
    "Model training completed."
)


# ---------------------------------------------------------
# 9. MAKE PREDICTIONS
# ---------------------------------------------------------

predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# 10. CALCULATE SUMMARY METRICS
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
# 11. DISPLAY RESULTS
# ---------------------------------------------------------

print(
    "\n--- ENHANCED MODEL RESULTS ---"
)

print(
    f"Accuracy: {accuracy:.3f}"
)

print(
    f"Macro F1: {macro_f1:.3f}"
)


# ---------------------------------------------------------
# 12. CLASSIFICATION REPORT
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
# 13. CONFUSION MATRIX
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