import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load files
model = joblib.load("final_migraine_xgb_model.joblib")
encoder = joblib.load("migraine_label_encoder.joblib")
feature_columns = joblib.load("feature_columns.joblib")

# Page config
st.set_page_config(page_title="Migraine Type Classifier", layout="centered")
st.title("🧠 Migraine Type Classifier")
st.markdown("This tool predicts the likely type of migraine based on reported symptoms. Use this as a digital triage assistant.")

# Sidebar: input form
st.sidebar.header("📋 Patient Info")
age = st.sidebar.slider("Age", 10, 80, 30)
duration = st.sidebar.selectbox("Attack Duration (hours)", [1, 2, 3])
frequency = st.sidebar.slider("Attack Frequency per Month", 1, 10, 3)

st.sidebar.header("🧠 Pain Description")
location = st.sidebar.selectbox("Pain Location", ["Unilateral", "Bilateral", "Frontal", "Temporal"])
character = st.sidebar.selectbox("Pain Character", ["Throbbing", "Pressing", "Sharp", "Dull"])

st.sidebar.header("🧾 Additional Symptoms")
symptom_list = ['Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory',
                'Dysphasia', 'Dysarthria', 'Vertigo', 'Tinnitus', 'Hypoacusis',
                'Diplopia', 'Defect', 'Ataxia', 'Conscience', 'Paresthesia', 'DPF']

symptom_data = {symptom: int(st.sidebar.checkbox(symptom)) for symptom in symptom_list}

# Encoding categorical variables
location_encoded = {
    'Location_2': 1 if location == 'Bilateral' else 0,
    'Location_3': 1 if location == 'Frontal' else 0,
    'Location_4': 1 if location == 'Temporal' else 0
}
character_encoded = {
    'Character_2': 1 if character == 'Pressing' else 0,
    'Character_3': 1 if character == 'Sharp' else 0,
    'Character_4': 1 if character == 'Dull' else 0
}

# Prepare input
input_dict = {
    'Age': age,
    'Duration': duration,
    'Frequency': frequency,
    **symptom_data,
    **location_encoded,
    **character_encoded
}

input_df = pd.DataFrame([input_dict])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# Prediction
if st.button("🔮 Predict Migraine Type"):
    pred_proba = model.predict_proba(input_df)
    prediction = np.argmax(pred_proba)
    confidence = round(np.max(pred_proba) * 100, 2)
    label = encoder.inverse_transform([prediction])[0]

    st.subheader("Result")
    st.success(f"🎯 Predicted Migraine Type: **{label}**")
    st.info(f"Confidence Score: {confidence}%")

st.markdown("---")
st.caption("📍 Built by Mohammed Masood Ahmed — NeuroHealthTech | 2025")
