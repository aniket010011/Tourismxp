import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# IMPORTANT: ensure model classes are available before loading
import lightgbm  # noqa: F401
import xgboost   # noqa: F401

st.set_page_config(page_title="Tourism Experience Predictor & Recommender", layout="wide")

# ==============================
# CACHED LOADERS
# ==============================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load trained pipelines and recommender assets."""
    reg_pipeline = joblib.load("artifacts/regression_pipeline.pkl")
    clf_pipeline = joblib.load("artifacts/classification_pipeline.pkl")

    # Optional recommender assets (safe load)
    try:
        user_item_matrix = joblib.load("artifacts/user_item_matrix.pkl")
        item_similarity = joblib.load("artifacts/item_similarity.pkl")
        rec_df = joblib.load("artifacts/recommender_df.pkl")
    except Exception:
        user_item_matrix, item_similarity, rec_df = None, None, None

    return reg_pipeline, clf_pipeline, user_item_matrix, item_similarity, rec_df


reg_pipeline, clf_pipeline, user_item_matrix, item_similarity, rec_df = load_artifacts()


# ==============================
# HELPER FUNCTIONS
# ==============================


def predict_rating(input_df: pd.DataFrame):
    return reg_pipeline.predict(input_df)[0]


def predict_investment(input_df: pd.DataFrame):
    return clf_pipeline.predict(input_df)[0]


def recommend_attractions(user_id, top_n=5):
    """Simple collaborative filtering recommender."""
    if user_item_matrix is None or item_similarity is None or rec_df is None:
        return pd.DataFrame({"Message": ["Recommender assets not available"]})

    if user_id not in user_item_matrix.index:
        return rec_df[["AttractionId", "Attraction"]].drop_duplicates().head(top_n)

    user_ratings = user_item_matrix.loc[user_id]
    scores = {}

    for item in user_item_matrix.columns:
        if user_ratings[item] > 0:
            continue

        weighted_sum = 0
        sim_sum = 0

        for rated_item in user_item_matrix.columns:
            rating = user_ratings[rated_item]
            if rating > 0:
                sim = item_similarity.loc[item, rated_item]
                if sim > 0:
                    weighted_sum += sim * rating
                    sim_sum += sim

        if sim_sum > 0:
            scores[item] = weighted_sum / sim_sum

    if len(scores) == 0:
        return rec_df[["AttractionId", "Attraction"]].drop_duplicates().head(top_n)

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_ids = [i[0] for i in top_items]

    return (
        rec_df[rec_df["AttractionId"].isin(top_ids)]
        [["AttractionId", "Attraction"]]
        .drop_duplicates()
        .head(top_n)
    )


# ==============================
# UI
# ==============================

st.title("🌍 Tourism Experience Analytics")

mode = st.sidebar.selectbox(
    "Choose Function",
    [
        "⭐ Rating Prediction",
        "💼 Investment Classification",
        "🎯 Attraction Recommendation",
    ],
)


# ==============================
# INPUT FORM (GENERIC)
# ==============================


def user_input_form():
    """Update fields to match your training features."""
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


# ==============================
# PAGES
# ==============================

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
    st.subheader("Personalized Attraction Recommendation")

    if user_item_matrix is None:
        st.warning("Recommender not available. Upload recommender artifacts.")
    else:
        user_id = st.number_input(
            "Enter User ID", int(user_item_matrix.index.min()), int(user_item_matrix.index.max()), int(user_item_matrix.index.min())
        )

        if st.button("Recommend"):
            try:
                recs = recommend_attractions(user_id, top_n=5)
                st.dataframe(recs, use_container_width=True)
            except Exception as e:
                st.error(f"Recommendation failed: {e}")


st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
