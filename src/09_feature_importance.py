import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

data = pd.read_csv(
    "data/processed/enhanced_incident_features.csv"
)


# ---------------------------------------------------------
# 2. FEATURES AND TARGET
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
# 4. FEATURE GROUPS
# ---------------------------------------------------------

categorical_features = [
    "location",
    "severity_type",
]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


# ---------------------------------------------------------
# 5. PREPROCESSING
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
# 6. RANDOM FOREST
# ---------------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


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
# 7. TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
)

print("Training completed.")


# ---------------------------------------------------------
# 8. GET TRANSFORMED FEATURE NAMES
# ---------------------------------------------------------

fitted_preprocessor = model.named_steps[
    "preprocessor"
]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


# ---------------------------------------------------------
# 9. GET RANDOM FOREST IMPORTANCES
# ---------------------------------------------------------

fitted_classifier = model.named_steps[
    "classifier"
]

importances = (
    fitted_classifier.feature_importances_
)


# ---------------------------------------------------------
# 10. CREATE FEATURE IMPORTANCE TABLE
# ---------------------------------------------------------

importance_table = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances,
    }
)


importance_table = (
    importance_table
    .sort_values(
        by="importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# 11. DISPLAY TOP 25 FEATURES
# ---------------------------------------------------------

print(
    "\n--- TOP 25 MOST IMPORTANT FEATURES ---"
)

print(
    importance_table.head(25).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 12. SAVE RESULTS
# ---------------------------------------------------------

importance_table.to_csv(
    "data/processed/feature_importance.csv",
    index=False,
)

print(
    "\nFeature importance saved successfully."
)