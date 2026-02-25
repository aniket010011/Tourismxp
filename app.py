import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Tourism Experience Analytics")

# ========================================
# LOAD ARTIFACTS
# ========================================
@st.cache_resource
def load_artifacts():
    reg_pipeline = joblib.load("artifacts/regression_pipeline.pkl")
    clf_pipeline = joblib.load("artifacts/classification_pipeline.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")

    # recommender artifacts
    rec_df = joblib.load("artifacts/rec_df.pkl")
    item_features = joblib.load("artifacts/item_features.pkl")
    item_id_to_pos = joblib.load("artifacts/item_id_to_pos.pkl")

    return (
        reg_pipeline,
        clf_pipeline,
        label_encoder,
        rec_df,
        item_features,
        item_id_to_pos,
    )


(
    reg_pipeline,
    clf_pipeline,
    label_encoder,
    rec_df,
    item_features,
    item_id_to_pos,
) = load_artifacts()

# ========================================
# SIDEBAR NAVIGATION
# ========================================
st.sidebar.header("Choose Function")

page = st.sidebar.selectbox(
    "Select Module",
    ["⭐ ML Prediction", "🎯 Recommendation System"]
)

# ============================================================
# PAGE 1 — ML PREDICTION
# ============================================================
if page == "⭐ ML Prediction":

    st.subheader("Predict Expected Rating")

    col1, col2, col3 = st.columns(3)

    with col1:
        visit_year = st.number_input("Visit Year", 2010, 2030, 2024)
        visit_month = st.number_input("Visit Month", 1, 12, 6)

    with col2:
        country = st.text_input("Country", "Indonesia")
        region = st.text_input("Region", "Asia")

    with col3:
        attraction_type = st.text_input("Attraction Type", "Temple")

    if st.button("Predict"):

        try:
            input_df = pd.DataFrame({
                "VisitYear": [visit_year],
                "VisitMonth": [visit_month],
                "Country": [country],
                "Region": [region],
                "AttractionType": [attraction_type],
            })

            # regression
            rating_pred = reg_pipeline.predict(input_df)[0]

            # classification
            mode_encoded = clf_pipeline.predict(input_df)[0]
            mode_label = label_encoder.inverse_transform([mode_encoded])[0]

            st.success(f"⭐ Predicted Rating: {rating_pred:.2f}")
            st.info(f"🧭 Predicted Visit Mode: {mode_label}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ============================================================
# PAGE 2 — RECOMMENDER
# ============================================================
elif page == "🎯 Recommendation System":

    st.subheader("Attraction Recommender")

    # ----------------------------------------
    # build attraction list safely
    # ----------------------------------------
    attraction_list = sorted(rec_df["Attraction"].dropna().unique())

    selected_item = st.selectbox(
        "Choose an attraction you liked",
        attraction_list
    )

    top_n = st.slider("Number of recommendations", 3, 10, 5)

    # ----------------------------------------
    # cosine similarity recommender
    # ----------------------------------------
    def recommend_items(item_name, top_n=5):
        if item_name not in rec_df["Attraction"].values:
            return []

        item_id = rec_df.loc[
            rec_df["Attraction"] == item_name,
            "AttractionId"
        ].iloc[0]

        pos = item_id_to_pos[item_id]

        query_vec = item_features[pos].reshape(1, -1)
        scores = item_features @ query_vec.T
        scores = scores.flatten()

        top_indices = np.argsort(scores)[::-1][1: top_n + 1]

        recommended_ids = [
            list(item_id_to_pos.keys())[i] for i in top_indices
        ]

        recs = rec_df[
            rec_df["AttractionId"].isin(recommended_ids)
        ]["Attraction"].unique().tolist()

        return recs

    # ----------------------------------------
    # button
    # ----------------------------------------
    if st.button("Get Recommendations"):

        try:
            recs = recommend_items(selected_item, top_n)

            if len(recs) == 0:
                st.warning("No recommendations found.")
            else:
                st.success("✨ Recommended Attractions")

                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")

        except Exception as e:
            st.error(f"Recommendation failed: {e}")
