import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

ARTIFACT_DIR = Path("artifacts")

# ================================
# LOAD MODELS (CACHED)
# ================================
@st.cache_resource
def load_artifacts():
    regression_pipeline = joblib.load(ARTIFACT_DIR / "regression_pipeline.pkl")
    classification_pipeline = joblib.load(ARTIFACT_DIR / "classification_pipeline.pkl")

    # recommender artifacts
    try:
        rec_df = joblib.load(ARTIFACT_DIR / "rec_df.pkl")
        item_features = joblib.load(ARTIFACT_DIR / "item_features.pkl")
        item_id_to_pos = joblib.load(ARTIFACT_DIR / "item_id_to_pos.pkl")
        tfidf = joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.pkl")
    except Exception:
        rec_df = None
        item_features = None
        item_id_to_pos = None
        tfidf = None

    return (
        regression_pipeline,
        classification_pipeline,
        rec_df,
        item_features,
        item_id_to_pos,
        tfidf,
    )


(
    regression_pipeline,
    classification_pipeline,
    rec_df,
    item_features,
    item_id_to_pos,
    tfidf,
) = load_artifacts()

# ================================
# SIDEBAR NAVIGATION
# ================================
st.sidebar.title("⚙️ Choose Function")

page = st.sidebar.selectbox(
    "Select Page",
    ["⭐ ML Prediction", "🎯 Recommendation System"]
)

# =====================================================
# PAGE 1 — ML PREDICTION
# =====================================================
if page == "⭐ ML Prediction":

    st.title("🌍 Tourism Experience Analytics")
    st.subheader("Predict Expected Rating")

    col1, col2, col3 = st.columns(3)

    with col1:
        visit_year = st.number_input("Visit Year", 2015, 2035, 2024)
        visit_month = st.number_input("Visit Month", 1, 12, 6)

    with col2:
        country = st.selectbox(
            "Country",
            ["Indonesia", "India", "USA", "UK", "Australia"]
        )
        region = st.selectbox(
            "Region",
            ["Asia", "Europe", "America", "Africa", "Oceania"]
        )

    with col3:
        attraction_type = st.selectbox(
            "Attraction Type",
            ["Temple", "Beach", "Museum", "Park", "Monument"]
        )

    if st.button("Predict Rating"):

        try:
            input_df = pd.DataFrame([{
                "VisitYear": visit_year,
                "VisitMonth": visit_month,
                "Country": country,
                "Region": region,
                "AttractionType": attraction_type,
            }])

            # regression
            rating_pred = regression_pipeline.predict(input_df)[0]

            # classification
            visit_mode_pred = classification_pipeline.predict(input_df)[0]

            st.success(f"⭐ Predicted Rating: {rating_pred:.2f}")
            st.info(f"🧭 Predicted Visit Mode: {visit_mode_pred}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =====================================================
# PAGE 2 — RECOMMENDER
# =====================================================
elif page == "🎯 Recommendation System":

    st.title("🎯 Attraction Recommendation System")

    if rec_df is None:
        st.error("❌ Recommender artifacts not found.")
        st.stop()

    attraction_list = sorted(rec_df["Attraction"].unique())

    selected_attraction = st.selectbox(
        "Select an attraction you like",
        attraction_list
    )

    top_n = st.slider("Number of recommendations", 3, 10, 5)

    # ----------------------------
    # RECOMMEND FUNCTION
    # ----------------------------
    def recommend_items(attraction_name, top_n=5):

        if attraction_name not in item_id_to_pos:
            return []

        idx = item_id_to_pos[attraction_name]

        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(
            item_features[idx],
            item_features
        ).flatten()

        similar_indices = sims.argsort()[::-1][1:top_n + 1]

        return rec_df.iloc[similar_indices][
            ["Attraction", "Country", "Region", "AttractionType"]
        ]

    # ----------------------------
    # BUTTON
    # ----------------------------
    if st.button("Get Recommendations"):

        try:
            recommendations = recommend_items(selected_attraction, top_n)

            if len(recommendations) == 0:
                st.warning("No recommendations found.")
            else:
                st.success("✨ Recommended Attractions")
                st.dataframe(recommendations, use_container_width=True)

        except Exception as e:
            st.error(f"Recommendation failed: {e}")

# ================================
# FOOTER
# ================================
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")
