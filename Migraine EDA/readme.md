# 🧠 Migraine Type Classification App

This Streamlit web app predicts the type of migraine a patient might be experiencing based on symptom data. It's designed to support clinical triage and digital decision-support tools in healthcare.

---

## 🚀 Project Overview

Doctors often struggle to differentiate between migraine types due to overlapping symptoms and subjective reporting. This app uses a machine learning model trained on patient-reported data to assist in the early identification of migraine types.

### ✅ Key Features:
- Predicts migraine type from symptoms
- Interactive Streamlit UI
- SHAP-based explainability
- Trained on real migraine episode dataset
- Portable and ready for deployment

---

## 📊 Input Features

- **Age**, **Duration**, **Frequency**
- **Pain Location**: Unilateral, Bilateral, Frontal, Temporal
- **Pain Character**: Throbbing, Pressing, Sharp, Dull
- **Symptoms** (checkboxes):
  - Nausea, Vomit, Phonophobia, Photophobia, Visual, Sensory
  - Dysphasia, Dysarthria, Vertigo, Tinnitus, Hypoacusis
  - Diplopia, Defect, Ataxia, Conscience, Paresthesia, DPF

---

## 📦 Files Included

| File Name                      | Description                                 |
|-------------------------------|---------------------------------------------|
| `app.py`                      | Streamlit frontend                          |
| `final_migraine_xgb_model.joblib` | Trained XGBoost model                  |
| `migraine_label_encoder.joblib` | Label encoder for migraine types        |
| `feature_columns.joblib`      | Feature order used by the model             |
| `requirements.txt`            | Python dependencies                         |

---

## 🛠️ How to Run Locally

### 1. Clone the repo and navigate to the folder:
```bash
git clone https://github.com/YOUR_USERNAME/migraine-classifier.git
cd migraine-classifier
