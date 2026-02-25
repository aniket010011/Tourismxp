import streamlit as st
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Tourism Experience Analytics")
st.subheader("Predict Expected Rating")

# ===============================
# LOAD ARTIFACTS
# ===============================
@st.cache_resource
def load_models():
    reg_pipeline = joblib.load("artifacts/regression_pipeline.pkl")
    clf_pipeline = joblib.load("artifacts/classification_pipeline.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")
    return reg_pipeline, clf_pipeline, label_encoder

reg_pipeline, clf_pipeline, label_encoder = load_models()

# ===============================
# USER INPUTS
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    visit_year = st.number_input("Visit Year", 2010, 2030, 2024)
    visit_month = st.number_input("Visit Month", 1, 12, 6)

with col2:
    country = st.text_input("Country", "Indonesia")
    region = st.text_input("Region", "Asia")

with col3:
    attraction_type = st.text_input("Attraction Type", "Temple")

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("Predict Rating"):

    try:
        # ✅ EXACT columns expected by pipeline
        input_df = pd.DataFrame({
            "VisitYear": [visit_year],
            "VisitMonth": [visit_month],
            "Country": [country],
            "Region": [region],
            "AttractionType": [attraction_type],
        })

        # ===============================
        # REGRESSION
        # ===============================
        rating_pred = reg_pipeline.predict(input_df)[0]

        # ===============================
        # CLASSIFICATION
        # ===============================
        mode_encoded = clf_pipeline.predict(input_df)[0]
        mode_label = label_encoder.inverse_transform([mode_encoded])[0]

        # ===============================
        # DISPLAY
        # ===============================
        st.success(f"⭐ Predicted Rating: **{rating_pred:.2f}**")
        st.info(f"🧭 Predicted Visit Mode: **{mode_label}**")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
