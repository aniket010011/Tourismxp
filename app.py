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
# Recommender Function (FIXED)
# =====================================================
def recommend_items(attraction_name, top_n=5):
    try:
        # get AttractionId
        attraction_id = rec_df.loc[
            rec_df["Attraction"] == attraction_name,
            "AttractionId",
        ].values[0]

        # 🔥 CRITICAL FIX
        attraction_id = int(attraction_id)

        if attraction_id not in item_id_to_pos:
            return []

        idx = item_id_to_pos[attraction_id]

        similarity_scores = list(enumerate(item_features[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

        # fast reverse map
        pos_to_item = {v: k for k, v in item_id_to_pos.items()}

        recommendations = []
        for pos, score in similarity_scores:
            rec_item_id = pos_to_item.get(pos)
            if rec_item_id is None:
                continue

            rec_name = rec_df.loc[
                rec_df["AttractionId"] == rec_item_id,
                "Attraction",
            ]

            if len(rec_name) > 0:
                recommendations.append(rec_name.values[0])

        return recommendations

    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return []


# =====================================================
# Sidebar Navigation
# =====================================================
st.sidebar.title("Choose Function")
page = st.sidebar.selectbox(
    "",
    ["Rating Prediction", "Recommendation"],
)

# =====================================================
# Title
# =====================================================
st.title("🌍 Tourism Experience Analytics")

# =====================================================
# PAGE 1 — PREDICTION
# =====================================================
if page == "Rating Prediction":
    st.subheader("Predict Expected Rating")

    col1, col2, col3 = st.columns(3)

    with col1:
        visit_year = st.number_input("Visit Year", value=2024)
        visit_month = st.number_input("Visit Month", min_value=1, max_value=12, value=6)

    with col2:
        visit_mode = st.selectbox("Visit Mode", label_encoder.classes_)
        country = st.selectbox("Country", sorted(rec_df.get("Country", pd.Series(["Indonesia"])).unique()))

    with col3:
        region = st.selectbox("Region", sorted(rec_df.get("Region", pd.Series(["Asia"])).unique()))
        attraction_type = st.selectbox(
            "Attraction Type",
            sorted(rec_df.get("AttractionType", pd.Series(["Temple"])).unique()),
        )

    if st.button("Predict Rating"):
        try:
            input_df = pd.DataFrame(
                {
                    "VisitYear": [visit_year],
                    "VisitMonth": [visit_month],
                    "VisitMode": [visit_mode],
                    "Country": [country],
                    "Region": [region],
                    "AttractionType": [attraction_type],
                }
            )

            # regression
            rating_pred = regression_pipeline.predict(input_df)[0]

            # classification
            visit_mode_encoded = classification_pipeline.predict(input_df)[0]
            visit_mode_label = label_encoder.inverse_transform([visit_mode_encoded])[0]

            st.success(f"⭐ Predicted Rating: {rating_pred:.2f}")
            st.info(f"🧭 Predicted Visit Mode: {visit_mode_label}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =====================================================
# PAGE 2 — RECOMMENDATION
# =====================================================
else:
    st.subheader("Attraction Recommendation")

    attraction_list = sorted(rec_df["Attraction"].dropna().unique())

    selected_attraction = st.selectbox("Choose an attraction", attraction_list)
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Get Recommendations"):
        recs = recommend_items(selected_attraction, top_n)

        if not recs:
            st.warning("No recommendations found.")
        else:
            st.success("Recommended Attractions")
            for i, r in enumerate(recs, 1):
                st.write(f"{i}. {r}")
