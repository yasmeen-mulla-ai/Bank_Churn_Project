import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv(r'C:\Users\YASMEEN\OneDrive\Desktop\ML intern\Assignment\Bank_Customer_Churn\Churn_Modelling.csv')

print("\nDataset Shape")
print(df.shape)

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================
# DATA CLEANING
# ==========================

df.drop_duplicates(inplace=True)

# Remove unwanted columns
df.drop(
    columns=[
        'RowNumber',
        'CustomerId',
        'Surname'
    ],
    inplace=True
)

# ==========================
# FEATURE ENGINEERING
# ==========================

# Feature 1
df["BalanceSalaryRatio"] = (
    df["Balance"] /
    (df["EstimatedSalary"] + 1)
)

# Feature 2
df["ProductsPerTenure"] = (
    df["NumOfProducts"] /
    (df["Tenure"] + 1)
)

# Feature 3 (Binning)
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18,30,45,60,100],
    labels=[
        "Young",
        "Adult",
        "MiddleAge",
        "Senior"
    ]
)

# ==========================
# SPLIT DATA
# ==========================

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# PREPROCESSING
# ==========================

cat_cols = [
    'Geography',
    'Gender',
    'AgeGroup'
]

num_cols = [
    col for col in X.columns
    if col not in cat_cols
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            cat_cols
        )
    ]
)

# ==========================
# MODEL 1
# ==========================

lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)

lr_pred = lr_pipeline.predict(X_test)

# ==========================
# MODEL 2
# ==========================

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeClassifier())
])

dt_pipeline.fit(X_train, y_train)

dt_pred = dt_pipeline.predict(X_test)

# ==========================
# MODEL 3
# ==========================

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)

# ==========================
# EVALUATION FUNCTION
# ==========================

def evaluate_model(name, y_test, pred):

    print("\n")
    print("="*40)
    print(name)
    print("="*40)

    print(
        "Accuracy:",
        accuracy_score(y_test, pred)
    )

    print(
        "Precision:",
        precision_score(y_test, pred)
    )

    print(
        "Recall:",
        recall_score(y_test, pred)
    )

    print(
        "F1 Score:",
        f1_score(y_test, pred)
    )

    print(
        "ROC AUC:",
        roc_auc_score(y_test, pred)
    )

    print(
        "\nConfusion Matrix\n",
        confusion_matrix(y_test, pred)
    )

evaluate_model(
    "Logistic Regression",
    y_test,
    lr_pred
)

evaluate_model(
    "Decision Tree",
    y_test,
    dt_pred
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(
    rf_pipeline,
    "model.pkl"
)

joblib.dump(
    preprocessor,
    "preprocessor.pkl"
)

print("\nSaved Successfully")