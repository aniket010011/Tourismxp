import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Tourism Experience Analytics")

# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# ======================================================
# LOAD ARTIFACTS
# ======================================================
@st.cache_resource
def load_artifacts():
    try:
        reg_pipeline = joblib.load(ARTIFACT_DIR / "regression_pipeline.pkl")
        clf_pipeline = joblib.load(ARTIFACT_DIR / "classification_pipeline.pkl")
        label_encoder = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

    # --- recommender (optional safe load) ---
    def safe_load(name):
        path = ARTIFACT_DIR / name
        return joblib.load(path) if path.exists() else None

    item_features = safe_load("item_features.pkl")
    item_id_to_pos = safe_load("item_id_to_pos.pkl")
    rec_df = safe_load("rec_df.pkl")
    tfidf_vectorizer = safe_load("tfidf_vectorizer.pkl")

    return (
        reg_pipeline,
        clf_pipeline,
        label_encoder,
        item_features,
        item_id_to_pos,
        rec_df,
        tfidf_vectorizer,
    )


(
    reg_pipeline,
    clf_pipeline,
    label_encoder,
    item_features,
    item_id_to_pos,
    rec_df,
    tfidf_vectorizer,
) = load_artifacts()

# ======================================================
# SIDEBAR NAV
# ======================================================
page = st.sidebar.selectbox(
    "Choose Function",
    ["🔮 ML Prediction", "🎯 Recommendation"]
)

# ======================================================
# PAGE 1 — ML PREDICTION
# ======================================================
if page == "🔮 ML Prediction":

    st.subheader("Predict Expected Rating")

    col1, col2, col3 = st.columns(3)

    with col1:
        visit_year = st.number_input("Visit Year", 2010, 2035, 2024)
        visit_month = st.number_input("Visit Month", 1, 12, 6)

    with col2:
        country = st.text_input("Country", "Indonesia")
        region = st.text_input("Region", "Asia")

    with col3:
        attraction_type = st.text_input("Attraction Type", "Temple")

    if st.button("Predict Rating"):

        try:
            input_df = pd.DataFrame({
                "VisitYear": [visit_year],
                "VisitMonth": [visit_month],
                "Country": [country],
                "Region": [region],
                "AttractionType": [attraction_type],
            })

            # regression
            rating_pred = float(reg_pipeline.predict(input_df)[0])

            # classification
            mode_encoded = clf_pipeline.predict(input_df)[0]
            mode_label = label_encoder.inverse_transform([mode_encoded])[0]

            st.success(f"⭐ Predicted Rating: **{rating_pred:.2f}**")
            st.info(f"🧭 Predicted Visit Mode: **{mode_label}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ======================================================
# PAGE 2 — RECOMMENDATION
# ======================================================
elif page == "🎯 Recommendation":

    st.subheader("Attraction Recommendation")

    # ---------- HARD SAFETY ----------
    if rec_df is None or item_features is None:
        st.error("❌ Recommender artifacts missing.")
        st.stop()

    if "Attraction" not in rec_df.columns:
        st.error("❌ rec_df missing 'Attraction' column.")
        st.stop()

    # ---------- CLEAN DF ----------
    rec_df = rec_df.reset_index(drop=True)
    rec_df["Attraction"] = rec_df["Attraction"].astype(str)

    attraction_list = sorted(rec_df["Attraction"].unique())

    selected_attraction = st.selectbox(
        "Choose an attraction",
        attraction_list
    )

    top_n = st.slider("Number of recommendations", 3, 10, 5)

    # ======================================================
    # RECOMMENDER Function
    # ======================================================
    def recommend_items(attraction_name, top_n=5):
        try:
            # find index safely
            matches = rec_df.index[rec_df["Attraction"] == attraction_name]

            if len(matches) == 0:
                return []

            idx = int(matches[0])  

            # ensure numpy array
            features = np.array(item_features)

            # cosine similarity
            sim_scores = cosine_similarity(
                [features[idx]],
                features
            )[0]

            # get top indices
            similar_indices = np.argsort(sim_scores)[::-1][1: top_n + 1]

            recommendations = (
                rec_df.iloc[similar_indices]["Attraction"]
                .astype(str)
                .tolist()
            )

            return recommendations

        except Exception as e:
            st.error(f"Recommendation error: {e}")
            return []

    # ======================================================
    # BUTTON
    # ======================================================
    if st.button("Get Recommendations"):

        recs = recommend_items(selected_attraction, top_n)

        if recs:
            st.success("✨ Recommended Attractions:")
            for r in recs:
                st.write("👉", r)
        else:
            st.warning("No recommendations found.")
