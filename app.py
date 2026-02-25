import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Ensure model classes are available before loading
import lightgbm  # noqa: F401
import xgboost   # noqa: F401

st.set_page_config(
    page_title="Tourism Experience Predictor & Recommender",
    layout="wide"
)

# ======================================================
# LOAD ARTIFACTS
# ======================================================

@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Load all trained assets safely."""

    # Core ML pipelines
    reg_pipeline = joblib.load("artifacts/regression_pipeline.pkl")
    clf_pipeline = joblib.load("artifacts/classification_pipeline.pkl")

    # Recommender assets (optional but expected)
    try:
        rec_df = joblib.load("artifacts/rec_df.pkl")
        item_features = joblib.load("artifacts/item_features.pkl")
        item_id_to_pos = joblib.load("artifacts/item_id_to_pos.pkl")
        label_encoder = joblib.load("artifacts/label_encoder.pkl")
        tfidf_vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
    except Exception:
        rec_df = None
        item_features = None
        item_id_to_pos = None
        label_encoder = None
        tfidf_vectorizer = None

    return (
        reg_pipeline,
        clf_pipeline,
        rec_df,
        item_features,
        item_id_to_pos,
        label_encoder,
        tfidf_vectorizer,
    )


(
    reg_pipeline,
    clf_pipeline,
    rec_df,
    item_features,
    item_id_to_pos,
    label_encoder,
    tfidf_vectorizer,
) = load_artifacts()


# ======================================================
# HELPER FUNCTIONS
# ======================================================


def predict_rating(input_df: pd.DataFrame):
    return reg_pipeline.predict(input_df)[0]


def predict_investment(input_df: pd.DataFrame):
    return clf_pipeline.predict(input_df)[0]


def popularity_fallback(top_n=5):
    """Fallback recommender using popularity."""
    if rec_df is None:
        return pd.DataFrame({"Message": ["Recommender not available"]})

    popular = (
        rec_df.groupby("Attraction")["AttractionId"]
        .count()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return popular


# ======================================================
# UI
# ======================================================

st.title("🌍 Tourism Experience Analytics")

mode = st.sidebar.selectbox(
    "Choose Function",
    [
        "⭐ Rating Prediction",
        "💼 Investment Classification",
        "🎯 Attraction Recommendation",
    ],
)


# ======================================================
# INPUT FORM (EDIT TO MATCH YOUR TRAINING FEATURES)
# ======================================================


def user_input_form():
    col1, col2, col3 = st.columns(3)

    with col1:
        visit_year = st.number_input("Visit Year", 2010, 2035, 2024)
        visit_month = st.number_input("Visit Month", 1, 12, 6)

    with col2:
        visit_mode = st.text_input("Visit Mode", "Family")
        region = st.text_input("Region", "Asia")

    with col3:
        country = st.text_input("Country", "Indonesia")
        attraction_type = st.text_input("Attraction Type", "Temple")

    data = {
        "VisitYear": visit_year,
        "VisitMonth": visit_month,
        "VisitMode": visit_mode,
        "Region": region,
        "Country": country,
        "AttractionType": attraction_type,
    }

    return pd.DataFrame([data])


# ======================================================
# PAGES
# ======================================================

if mode == "⭐ Rating Prediction":
    st.subheader("Predict Expected Rating")

    input_df = user_input_form()

    if st.button("Predict Rating"):
        try:
            pred = predict_rating(input_df)
            st.success(f"Predicted Rating: {pred:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


elif mode == "💼 Investment Classification":
    st.subheader("Predict Investment Quality")

    input_df = user_input_form()

    if st.button("Predict Investment"):
        try:
            pred = predict_investment(input_df)
            st.success(f"Prediction: {pred}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


elif mode == "🎯 Attraction Recommendation":
    st.subheader("Popular Attractions")

    if rec_df is None:
        st.warning("Recommender assets not available.")
    else:
        top_n = st.slider("Number of recommendations", 3, 15, 5)

        if st.button("Show Recommendations"):
            try:
                recs = popularity_fallback(top_n=top_n)
                st.dataframe(recs, use_container_width=True)
            except Exception as e:
                st.error(f"Recommendation failed: {e}")


st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
