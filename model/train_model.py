import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _assign_risk(row):
    score = 0
    age = row["Age"]
    bp = row["Blood_Pressure"]
    hr = row["Heart_Rate"]
    temp = row["Temperature"]
    symptoms = row["Symptoms"]
    condition = row["Pre_Existing_Conditions"]

    if age >= 65:
        score += 2
    elif age >= 45:
        score += 1

    if bp >= 150:
        score += 2
    elif bp >= 130:
        score += 1

    if hr >= 120:
        score += 2
    elif hr >= 100:
        score += 1

    if temp >= 102.0:
        score += 2
    elif temp >= 100.4:
        score += 1

    if symptoms in {"Chest Pain", "Shortness of Breath"}:
        score += 2
    elif symptoms in {"Fever", "Dizziness"}:
        score += 1

    if condition in {"Heart Disease"}:
        score += 2
    elif condition in {"Hypertension", "Diabetes"}:
        score += 1

    score += np.random.choice([0, 0, 1])

    if score >= 7:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"


def main():
    n_samples = 1200
    genders = ["Male", "Female", "Other"]
    symptoms = [
        "Chest Pain",
        "Shortness of Breath",
        "Fever",
        "Headache",
        "Abdominal Pain",
        "Dizziness",
        "Cough",
        "Fatigue",
    ]
    conditions = ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"]

    data = pd.DataFrame(
        {
            "Age": np.random.randint(1, 90, size=n_samples),
            "Gender": np.random.choice(genders, size=n_samples),
            "Blood_Pressure": np.random.randint(90, 180, size=n_samples),
            "Heart_Rate": np.random.randint(55, 140, size=n_samples),
            "Temperature": np.round(np.random.uniform(97.0, 104.0, size=n_samples), 1),
            "Symptoms": np.random.choice(symptoms, size=n_samples),
            "Pre_Existing_Conditions": np.random.choice(conditions, size=n_samples),
        }
    )
    data["Risk_Level"] = data.apply(_assign_risk, axis=1)

    os.makedirs("data", exist_ok=True)
    data.to_csv(os.path.join("data", "synthetic_data.csv"), index=False)
    data.to_excel(os.path.join("data", "synthetic_data.xlsx"), index=False)

    X = data.drop(columns=["Risk_Level"])
    y = data["Risk_Level"]

    categorical_features = ["Gender", "Symptoms", "Pre_Existing_Conditions"]
    numeric_features = ["Age", "Blood_Pressure", "Heart_Rate", "Temperature"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print("\n✅ Classification Report:\n")
    print(classification_report(y_test, predictions))

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, os.path.join("model", "model.pkl"))
    print("\n✅ Model saved to model/model.pkl")


if __name__ == "__main__":
    main()
