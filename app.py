"""
AI Smart Patient Triage System
Professional Multi-Page Web Application
"""

import datetime as dt
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.prediction import predict_risk
from database import init_db, insert_patient, get_all_patients


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Smart Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database (cached)
@st.cache_resource
def initialize_database():
    init_db()

initialize_database()

# Cache database queries for performance
@st.cache_data(ttl=5)
def get_patients_from_db():
    return get_all_patients()


# ==========================================
# CONSTANTS & STYLING
# ==========================================
RISK_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}
RISK_ORDER = {"High": 3, "Medium": 2, "Low": 1}

# Custom CSS for hospital-style UI
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
    }
    .nav-button {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .nav-button:hover {
        transform: scale(1.05);
    }
    .risk-badge {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _init_state():
    """Initialize session state variables"""
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "patients" not in st.session_state:
        st.session_state.patients = []


def _display_logo():
    """Display hospital logo and header"""
    st.markdown("""
        <div class="main-header">
            <h1>🏥 AI Smart Patient Triage System</h1>
            <p>Intelligent Patient Risk Assessment & Department Routing</p>
        </div>
    """, unsafe_allow_html=True)


def _risk_badge(risk_label: str):
    """Display colored risk badge"""
    color = RISK_COLORS.get(risk_label, "#607d8b")
    st.markdown(
        f'<div class="risk-badge" style="background:{color};color:white;">'
        f'{risk_label} Risk</div>',
        unsafe_allow_html=True
    )


def _format_confidence(score: float) -> str:
    """Format confidence score as percentage"""
    return f"{max(0.0, min(1.0, score)) * 100:.1f}%"


def _generate_patient_id() -> str:
    """Generate unique patient ID"""
    return f"PT-{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[-10:]}"


def _save_to_database(input_data: Dict, prediction: Dict):
    """Save patient data and prediction to database"""
    db_data = input_data.copy()
    db_data.update(prediction)
    insert_patient(db_data)
    
    # Also add to session state
    record = {
        "patient_id": _generate_patient_id(),
        "submitted_at": dt.datetime.utcnow(),
        "symptoms": input_data.get("Symptoms"),
        "age": input_data.get("Age"),
        "gender": input_data.get("Gender"),
        "blood_pressure": input_data.get("Blood_Pressure"),
        "heart_rate": input_data.get("Heart_Rate"),
        "temperature": input_data.get("Temperature"),
        "conditions": input_data.get("Pre_Existing_Conditions"),
        "risk_label": prediction["risk_label"],
        "confidence": prediction["confidence_score"],
        "recommended_department": prediction["recommended_department"],
    }
    st.session_state.patients.append(record)


# ==========================================
# PAGE: HOME
# ==========================================
def page_home():
    """Home page with navigation"""
    _display_logo()
    
    st.markdown("### Welcome to the AI-Powered Healthcare System")
    st.markdown("""
    This system uses machine learning to:
    - **Assess patient risk levels** (High, Medium, Low)
    - **Recommend appropriate departments** for treatment
    - **Prioritize emergency cases** automatically
    - **Provide explainable AI insights** for medical staff
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background:#f0f8ff;padding:2rem;border-radius:10px;text-align:center;'>
                <h2>👤 Patient Portal</h2>
                <p style='font-size:1.1rem;'>Submit your symptoms and vitals for instant risk assessment</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Enter Patient Portal", key="btn_patient", use_container_width=True):
            st.session_state.page = "Patient Portal"
            st.rerun()
    
    with col2:
        st.markdown("""
            <div style='background:#fff0f5;padding:2rem;border-radius:10px;text-align:center;'>
                <h2>🏥 Hospital Dashboard</h2>
                <p style='font-size:1.1rem;'>Monitor patient queue, analytics, and risk distribution</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Open Dashboard", key="btn_dashboard", use_container_width=True):
            st.session_state.page = "Hospital Dashboard"
            st.rerun()
    
    st.markdown("---")
    
    # Quick Stats
    patients_db = get_patients_from_db()
    if patients_db:
        st.markdown("### 📈 System Statistics")
        df_stats = pd.DataFrame(patients_db, columns=[
            "ID", "Age", "Gender", "BP", "HR", "Temp",
            "Symptoms", "Condition", "Risk", "Confidence",
            "Department", "Created"
        ])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", len(df_stats))
        col2.metric("High Risk", len(df_stats[df_stats["Risk"] == "High"]), 
                   delta="🚨" if len(df_stats[df_stats["Risk"] == "High"]) > 0 else "✅")
        col3.metric("Medium Risk", len(df_stats[df_stats["Risk"] == "Medium"]))
        col4.metric("Low Risk", len(df_stats[df_stats["Risk"] == "Low"]))


# ==========================================
# PAGE: PATIENT PORTAL
# ==========================================
def page_patient_portal():
    """Patient portal for risk assessment"""
    st.markdown("## 👤 Patient Portal")
    st.markdown("#### Complete the form below for AI-powered risk assessment")
    
    st.markdown("---")
    
    with st.form("patient_assessment_form", clear_on_submit=False):
        # Two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📋 Basic Information")
            age = st.number_input("Age", min_value=0, max_value=120, value=45, help="Patient's age in years")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            st.markdown("##### 🩺 Vital Signs")
            blood_pressure = st.number_input(
                "Blood Pressure (mmHg)", 
                min_value=70, 
                max_value=200, 
                value=120,
                help="Systolic blood pressure"
            )
            heart_rate = st.number_input(
                "Heart Rate (bpm)", 
                min_value=40, 
                max_value=180, 
                value=80,
                help="Beats per minute"
            )
            temperature = st.number_input(
                "Temperature (°F)", 
                min_value=95.0, 
                max_value=105.0, 
                value=98.6,
                step=0.1,
                help="Body temperature in Fahrenheit"
            )
        
        with col2:
            st.markdown("##### 🤒 Symptoms")
            symptoms = st.multiselect(
                "Select all symptoms",
                [
                    "Chest Pain",
                    "Shortness of Breath",
                    "Fever",
                    "Headache",
                    "Abdominal Pain",
                    "Dizziness",
                    "Cough",
                    "Fatigue",
                ],
                default=["Fatigue"],
                help="Select one or more symptoms"
            )
            
            st.markdown("##### 🏥 Medical History")
            conditions = st.selectbox(
                "Pre-Existing Conditions",
                ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"],
                help="Select primary pre-existing condition"
            )
            
            st.markdown("##### 📄 Optional: Upload EHR")
            uploaded_file = st.file_uploader(
                "Upload Electronic Health Record",
                type=["pdf", "txt"],
                help="Optional: Upload medical records (PDF or TXT)"
            )
        
        st.markdown("---")
        col_submit, col_clear = st.columns([3, 1])
        with col_submit:
            submit_button = st.form_submit_button("🔍 Analyze Risk", use_container_width=True)
        with col_clear:
            clear_button = st.form_submit_button("🔄 Clear", use_container_width=True)
    
    # Process submission
    if submit_button:
        input_data = {
            "Age": age,
            "Gender": gender,
            "Blood_Pressure": blood_pressure,
            "Heart_Rate": heart_rate,
            "Temperature": temperature,
            "Symptoms": symptoms,
            "Pre_Existing_Conditions": conditions,
        }
        
        with st.spinner("🔍 Analyzing patient data using AI model..."):
            prediction = predict_risk(input_data)
        
        # Display file upload confirmation
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        st.markdown("---")
        st.markdown("## 📊 Assessment Results")
        
        # Risk Badge
        col_risk, col_conf, col_dept = st.columns(3)
        with col_risk:
            st.markdown("##### Risk Level")
            _risk_badge(prediction["risk_label"])
        
        with col_conf:
            st.markdown("##### Confidence Score")
            confidence = prediction["confidence_score"]
            st.progress(confidence)
            st.markdown(f"**{_format_confidence(confidence)}** confident")
        
        with col_dept:
            st.markdown("##### Recommended Department")
            st.info(f"**{prediction['recommended_department']}**")
        
        # Reason & Features
        st.markdown("---")
        col_reason, col_features = st.columns(2)
        
        with col_reason:
            st.markdown("##### 💡 Assessment Reason")
            st.markdown(f"> {prediction.get('department_reason', 'Based on current symptoms and vitals')}")
        
        with col_features:
            st.markdown("##### 🔍 Top Contributing Factors")
            features_df = pd.DataFrame(
                prediction["top_features"],
                columns=["Factor", "Contribution"]
            )
            st.dataframe(features_df, use_container_width=True, hide_index=True)
        
        # Alert for high risk
        if prediction["risk_label"] == "High":
            st.error("⚠️ **HIGH RISK DETECTED** - Please seek immediate medical attention!")
            st.toast("🚨 High risk patient - immediate care required!", icon="⚠️")
        
        # Save to database
        _save_to_database(input_data, prediction)
        st.success("✅ Assessment saved to hospital database")


# ==========================================
# PAGE: HOSPITAL DASHBOARD
# ==========================================
def page_hospital_dashboard():
    """Hospital dashboard with analytics"""
    st.markdown("## 🏥 Hospital Operations Dashboard")
    st.markdown("#### Real-time patient monitoring and analytics")
    
    # Refresh button
    col_title, col_refresh = st.columns([5, 1])
    with col_refresh:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Get data from database
    patients_db = get_patients_from_db()
    
    if not patients_db:
        st.info("📭 No patients in the system yet. Start by adding patients through the Patient Portal.")
        return
    
    # Convert to DataFrame
    df_patients = pd.DataFrame(patients_db, columns=[
        "ID", "Age", "Gender", "BP", "HR", "Temp",
        "Symptoms", "Condition", "Risk", "Confidence",
        "Department", "Created"
    ])
    
    # ==========================================
    # KEY METRICS
    # ==========================================
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_patients = len(df_patients)
    high_risk = len(df_patients[df_patients["Risk"] == "High"])
    medium_risk = len(df_patients[df_patients["Risk"] == "Medium"])
    low_risk = len(df_patients[df_patients["Risk"] == "Low"])
    
    col1.metric("Total Patients", total_patients, delta=None)
    col2.metric("🚨 Emergency (High)", high_risk, 
               delta=f"{(high_risk/total_patients*100):.1f}%" if total_patients > 0 else "0%",
               delta_color="inverse")
    col3.metric("⚠️ Medium Risk", medium_risk,
               delta=f"{(medium_risk/total_patients*100):.1f}%" if total_patients > 0 else "0%")
    col4.metric("✅ Low Risk", low_risk,
               delta=f"{(low_risk/total_patients*100):.1f}%" if total_patients > 0 else "0%",
               delta_color="normal")
    
    st.markdown("---")
    
    # ==========================================
    # ANALYTICS CHARTS
    # ==========================================
    st.markdown("### 📈 Analytics & Insights")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Risk Distribution")
        risk_counts = df_patients["Risk"].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0
        )
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
            hole=0.4
        )
        fig_risk.update_layout(showlegend=True, height=300)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col_chart2:
        st.markdown("#### Department Load")
        dept_counts = df_patients["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig_dept = px.bar(
            dept_counts,
            x="Department",
            y="Count",
            color="Department",
            text="Count"
        )
        fig_dept.update_layout(showlegend=False, height=300)
        fig_dept.update_traces(textposition='outside')
        st.plotly_chart(fig_dept, use_container_width=True)
    
    st.markdown("---")
    
    # ==========================================
    # PRIORITY QUEUE
    # ==========================================
    st.markdown("### 🎯 Priority Patient Queue")
    st.markdown("*Sorted by risk level (High → Medium → Low) and recent submissions*")
    
    # Sort by risk priority
    df_sorted = df_patients.copy()
    df_sorted["risk_order"] = df_sorted["Risk"].map(RISK_ORDER)
    df_sorted = df_sorted.sort_values(by=["risk_order", "Created"], ascending=[False, False])
    
    # Display with styling
    def highlight_risk(row):
        if row["Risk"] == "High":
            return ['background-color: #ffebee'] * len(row)
        elif row["Risk"] == "Medium":
            return ['background-color: #fff9e6'] * len(row)
        else:
            return ['background-color: #f1f8f4'] * len(row)
    
    styled_df = df_sorted.style.apply(highlight_risk, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ==========================================
    # DETAILED STATS
    # ==========================================
    st.markdown("### 📋 Detailed Statistics")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.markdown("#### Age Distribution")
        fig_age = px.histogram(df_patients, x="Age", nbins=10, color_discrete_sequence=['#667eea'])
        fig_age.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col_stat2:
        st.markdown("#### Gender Breakdown")
        gender_counts = df_patients["Gender"].value_counts()
        fig_gender = px.bar(x=gender_counts.index, y=gender_counts.values, color_discrete_sequence=['#764ba2'])
        fig_gender.update_layout(height=250, showlegend=False, xaxis_title="Gender", yaxis_title="Count")
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col_stat3:
        st.markdown("#### Average Confidence")
        avg_conf = df_patients["Confidence"].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_conf * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#667eea"},
                   'steps': [
                       {'range': [0, 50], 'color': "#ffebee"},
                       {'range': [50, 80], 'color': "#fff9e6"},
                       {'range': [80, 100], 'color': "#f1f8f4"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)


# ==========================================
# NAVIGATION & MAIN APP
# ==========================================
