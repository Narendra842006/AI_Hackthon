import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


# ======================================================
# MODEL PATH
# ======================================================
MODEL_PATH = os.path.join("model", "model.pkl")


# ======================================================
# DEFAULT VALUES
# ======================================================
DEFAULTS = {
    "Age": 45,
    "Gender": "Other",
    "Blood_Pressure": 120,
    "Heart_Rate": 80,
    "Temperature": 98.6,
    "Symptoms": "Fatigue",
    "Pre_Existing_Conditions": "None",
}


# ======================================================
# SYMPTOM SEVERITY (Used for normalization)
# ======================================================
SYMPTOM_SEVERITY = {
    "Chest Pain": 3,
    "Shortness of Breath": 3,
    "Fever": 2,
    "Dizziness": 2,
    "Abdominal Pain": 2,
    "Cough": 1,
    "Headache": 1,
    "Fatigue": 1,
}


# ======================================================
# BASE DEPARTMENT MAP
# ======================================================
DEPARTMENT_MAP = {
    "Chest Pain": "Cardiology",
    "Shortness of Breath": "Pulmonology",
    "Fever": "Infectious Disease",
    "Headache": "Neurology",
    "Abdominal Pain": "Gastroenterology",
    "Dizziness": "Emergency",
    "Cough": "Pulmonology",
    "Fatigue": "General Medicine",
}


# ======================================================
# LOAD MODEL (LOAD ONCE FOR PERFORMANCE)
# ======================================================
_pipeline = None


def _load_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run model/train_model.py first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


# ======================================================
# NORMALIZE SYMPTOMS
# ======================================================
def _normalize_symptoms(symptoms: List[str] | str | None) -> str:
    if symptoms is None:
        return DEFAULTS["Symptoms"]

    if isinstance(symptoms, str):
        return symptoms

    if not symptoms:
        return DEFAULTS["Symptoms"]

    # Pick highest severity symptom
    return max(symptoms, key=lambda item: SYMPTOM_SEVERITY.get(item, 0))


# ======================================================
# PREPARE INPUT DATAFRAME
# ======================================================
def _prepare_input(input_data: Dict) -> pd.DataFrame:
    payload = DEFAULTS.copy()
    payload.update({k: v for k, v in input_data.items() if v not in (None, "")})
    payload["Symptoms"] = _normalize_symptoms(payload.get("Symptoms"))
    return pd.DataFrame([payload])


# ======================================================
# FEATURE NAMES FOR EXPLAINABILITY
# ======================================================
def _get_feature_names(pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocess"]

    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
        ["Gender", "Symptoms", "Pre_Existing_Conditions"]
    )

    num_features = ["Age", "Blood_Pressure", "Heart_Rate", "Temperature"]

    return list(cat_features) + num_features


# ======================================================
# TOP CONTRIBUTING FEATURES
# ======================================================
def _top_features(
    pipeline, input_df: pd.DataFrame, top_n: int = 4
) -> List[Tuple[str, float]]:
    model = pipeline.named_steps["model"]
    transformed = pipeline.named_steps["preprocess"].transform(input_df)

    feature_names = _get_feature_names(pipeline)
    importances = model.feature_importances_

    if hasattr(transformed, "toarray"):
        values = transformed.toarray()[0]
    else:
        values = np.asarray(transformed)[0]

    contributions = importances * np.abs(values)

    top_idx = np.argsort(contributions)[-top_n:][::-1]

    return [(feature_names[i], float(contributions[i])) for i in top_idx]


# ======================================================
# MAIN PREDICTION FUNCTION
# ======================================================
def predict_risk(input_data: Dict) -> Dict:
    pipeline = _load_model()
    input_df = _prepare_input(input_data)

    # Predict probabilities
    proba = pipeline.predict_proba(input_df)[0]
    classes = pipeline.classes_

    best_idx = int(np.argmax(proba))
    risk_label = str(classes[best_idx])
    confidence_score = float(proba[best_idx])

    # Extract input values
    symptom = input_df.loc[0, "Symptoms"]
    condition = input_df.loc[0, "Pre_Existing_Conditions"]

    # ==================================================
    # ADVANCED DEPARTMENT ROUTING LOGIC
    # ==================================================

    # Base mapping
    recommended_department = DEPARTMENT_MAP.get(
        symptom, "General Medicine"
    )

    department_reason = f"Based on primary symptom: {symptom}"

    # High risk override → Emergency
    if risk_label == "High":
        recommended_department = "Emergency"
        department_reason = "High risk detected. Immediate emergency care required."

    # Cardiac override
    if symptom == "Chest Pain" or condition == "Heart Disease":
        if risk_label in ["High", "Medium"]:
            recommended_department = "Cardiology"
            department_reason = (
                "Cardiac-related symptom or condition detected."
            )

    # Respiratory override
    if symptom == "Shortness of Breath":
        recommended_department = "Pulmonology"
        department_reason = "Respiratory distress symptom detected."

    # Confidence-based escalation
    if confidence_score > 0.90 and risk_label == "High":
        recommended_department = "Emergency"
        department_reason = (
            "Very high confidence high-risk prediction. Escalated to Emergency."
        )

    # ==================================================
    # EXPLAINABILITY
    # ==================================================
    top_features = _top_features(pipeline, input_df)

    return {
        "risk_label": risk_label,
        "confidence_score": confidence_score,
        "recommended_department": recommended_department,
        "department_reason": department_reason,
        "top_features": top_features,
    }


# ======================================================
# TEST / MANUAL RUN
# ======================================================
if __name__ == "__main__":
    print("🏥 Testing Prediction Module...\n")
    
    sample_input = {
        "Age": 75,
        "Gender": "Male",
        "Blood_Pressure": 170,
        "Heart_Rate": 135,
        "Temperature": 103.5,
        "Symptoms": "Chest Pain",
        "Pre_Existing_Conditions": "Heart Disease",
    }

    result = predict_risk(sample_input)

    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Risk Level:       {result['risk_label']}")
    print(f"Confidence:       {result['confidence_score']:.2%}")
    print(f"Department:       {result['recommended_department']}")
    print(f"Reason:           {result['department_reason']}")
    print("\nTop Contributing Features:")
    for feature, value in result['top_features']:
        print(f"  • {feature}: {value:.4f}")
    print("=" * 60)
