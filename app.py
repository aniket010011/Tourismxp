import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

# =====================================================
# Paths
# =====================================================
ARTIFACT_DIR = Path("artifacts")
DATA_PATH = Path("clean_master_tourism_data.csv")

# =====================================================
# Load Artifacts
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        classification_pipeline = joblib.load(ARTIFACT_DIR / "classification_pipeline.pkl")
        regression_pipeline = joblib.load(ARTIFACT_DIR / "regression_pipeline.pkl")
        label_encoder = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")

        # recommender
        item_features = joblib.load(ARTIFACT_DIR / "item_features.pkl")
        item_id_to_pos = joblib.load(ARTIFACT_DIR / "item_id_to_pos.pkl")
        rec_df = joblib.load(ARTIFACT_DIR / "rec_df.pkl")

        return (
            classification_pipeline,
            regression_pipeline,
            label_encoder,
            item_features,
            item_id_to_pos,
            rec_df,
        )
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

(
    classification_pipeline,
    regression_pipeline,
    label_encoder,
    item_features,
    item_id_to_pos,
    rec_df,
) = load_artifacts()

# =====================================================
# Recommender Function
# =====================================================
def get_recommendations(attraction_name, top_n=5):
    try:
        # --- STEP 1: name → AttractionId ---
        match = rec_df[rec_df["Attraction"] == attraction_name]

        if match.empty:
            st.warning("Attraction not found in data.")
            return []

        attraction_id = match["AttractionId"].iloc[0]

        # --- STEP 2: AttractionId → position ---
        if attraction_id not in item_id_to_pos:
            st.warning("Attraction ID not in similarity matrix.")
            return []

        pos = item_id_to_pos[attraction_id]

        # --- STEP 3: similarity ---
        sim_scores = cosine_similarity(
            item_features[pos].reshape(1, -1),
            item_features
        ).flatten()

        # remove itself
        sim_scores[pos] = -1

        top_indices = sim_scores.argsort()[-top_n:][::-1]

        # --- STEP 4: map back to names ---
        reverse_map = {v: k for k, v in item_id_to_pos.items()}

        recommended_ids = [reverse_map[i] for i in top_indices]

        recommendations = (
            rec_df[rec_df["AttractionId"].isin(recommended_ids)]
            ["Attraction"]
            .drop_duplicates()
            .tolist()
        )

        return recommendations

    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return []
