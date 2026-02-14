import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "triage.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create patients table if it doesn't exist."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            created TEXT,
            age INTEGER,
            gender TEXT,
            bp INTEGER,
            hr INTEGER,
            temp REAL,
            symptoms TEXT,
            condition TEXT,
            risk TEXT,
            confidence REAL,
            department TEXT,
            extra JSON
        )
        """
    )
    conn.commit()
    conn.close()


def insert_patient(data: dict):
    """Insert a patient record. `data` is a dict with keys expected by the app."""
    conn = _get_conn()
    cur = conn.cursor()

    patient_id = data.get("patient_id") or f"PT-{int(datetime.utcnow().timestamp()*1000)}"
    created = data.get("created") or datetime.utcnow().isoformat()

    age = data.get("Age")
    gender = data.get("Gender")
    bp = data.get("Blood_Pressure")
    hr = data.get("Heart_Rate")
    temp = data.get("Temperature")
    symptoms = json.dumps(data.get("Symptoms"))
    condition = data.get("Pre_Existing_Conditions")
    risk = data.get("risk_label") or data.get("Risk")
    confidence = data.get("confidence_score") or data.get("Confidence")
    department = data.get("recommended_department") or data.get("Department")

    extra = {k: v for k, v in data.items() if k not in [
        "patient_id", "created", "Age", "Gender", "Blood_Pressure",
        "Heart_Rate", "Temperature", "Symptoms", "Pre_Existing_Conditions",
        "risk_label", "confidence_score", "recommended_department",
    ]}

    cur.execute(
        """
        INSERT OR REPLACE INTO patients
        (id, created, age, gender, bp, hr, temp, symptoms, condition, risk, confidence, department, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            created,
            age,
            gender,
            bp,
            hr,
            temp,
            symptoms,
            condition,
            risk,
            confidence,
            department,
            json.dumps(extra),
        ),
    )
    conn.commit()
    conn.close()


def get_all_patients():
    """Return a list of patient rows formatted for the app display."""
    conn = _get_conn()
    cur = conn.cursor()
    # Check schema in case older DB lacks 'created' column
    cur.execute("PRAGMA table_info(patients)")
    cols_info = cur.fetchall()
    cols = [c[1] for c in cols_info]
    if "created" in cols:
        cur.execute("SELECT * FROM patients ORDER BY created DESC")
    else:
        cur.execute("SELECT * FROM patients ORDER BY rowid DESC")
    rows = cur.fetchall()
    result = []
    for r in rows:
        symptoms = json.loads(r["symptoms"]) if r.get("symptoms") else []
        created_val = None
        if "created" in r.keys():
            created_val = r["created"]
        else:
            # try to extract from extra JSON if present
            try:
                extra = json.loads(r["extra"]) if r.get("extra") else {}
                created_val = extra.get("created")
            except Exception:
                created_val = None

        result.append([
            r["id"],
            r["age"],
            r["gender"],
            r["bp"],
            r["hr"],
            r["temp"],
            ", ".join(symptoms) if isinstance(symptoms, list) else symptoms,
            r["condition"],
            r["risk"],
            r["confidence"],
            r["department"],
            created_val,
        ])
    conn.close()
    return result
