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

# ===============================
# SIDEBAR NAVIGATION
# ===============================
page = st.sidebar.selectbox(
    "Choose Function",
    ["🔮 ML Prediction", "🎯 Recommendation"]
)

# ===============================
# LOAD ARTIFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    reg_pipeline = joblib.load("artifacts/regression_pipeline.pkl")
    clf_pipeline = joblib.load("artifacts/classification_pipeline.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")

    # recommender artifacts (safe load)
    try:
        item_features = joblib.load("artifacts/item_features.pkl")
        item_id_to_pos = joblib.load("artifacts/item_id_to_pos.pkl")
        rec_df = joblib.load("artifacts/rec_df.pkl")
        tfidf_vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")
    except:
        item_features = None
        item_id_to_pos = None
        rec_df = None
        tfidf_vectorizer = None

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

# =========================================================
# PAGE 1 — ML PREDICTION
# =========================================================
if page == "🔮 ML Prediction":

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

    if st.button("Predict Rating"):

        try:
            input_df = pd.DataFrame({
                "VisitYear": [visit_year],
                "VisitMonth": [visit_month],
                "Country": [country],
                "Region": [region],
                "AttractionType": [attraction_type],
            })

            # Regression
            rating_pred = reg_pipeline.predict(input_df)[0]

            # Classification
            mode_encoded = clf_pipeline.predict(input_df)[0]
            mode_label = label_encoder.inverse_transform([mode_encoded])[0]

            st.success(f"⭐ Predicted Rating: **{rating_pred:.2f}**")
            st.info(f"🧭 Predicted Visit Mode: **{mode_label}**")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# =========================================================
# PAGE 2 — RECOMMENDATION
# =========================================================
elif page == "🎯 Recommendation":

    st.subheader("Attraction Recommendation")

    # safety check
    if rec_df is None or item_features is None or item_id_to_pos is None:
        st.warning("⚠️ Recommender artifacts not found in /artifacts")
        st.stop()

    attraction_list = sorted(rec_df["Attraction"].unique())

    selected_attraction = st.selectbox(
        "Choose an attraction",
        attraction_list
    )

    top_n = st.slider("Number of recommendations", 3, 10, 5)

    # -----------------------------
    # CORRECT RECOMMENDER FUNCTION
    # -----------------------------
    def recommend_items(attraction_name, top_n=5):
        try:
            # ✅ Step 1: get AttractionId
            attraction_id = rec_df.loc[
                rec_df["Attraction"] == attraction_name,
                "AttractionId"
            ].values[0]

            # ✅ Step 2: map to matrix position
            if attraction_id not in item_id_to_pos:
                st.error("Attraction not found in similarity matrix.")
                return []

            idx = item_id_to_pos[attraction_id]

            # ✅ Step 3: get similarity row
            similarity_scores = list(enumerate(item_features[idx]))

            # ✅ Step 4: sort
            similarity_scores = sorted(
                similarity_scores,
                key=lambda x: x[1],
                reverse=True
            )[1: top_n + 1]

            # ✅ Step 5: map back to attractions
            recommendations = []

            for pos, score in similarity_scores:
                rec_item_id = list(item_id_to_pos.keys())[list(item_id_to_pos.values()).index(pos)]

                rec_name = rec_df.loc[
                    rec_df["AttractionId"] == rec_item_id,
                    "Attraction"
                ].values

                if len(rec_name) > 0:
                    recommendations.append(rec_name[0])

            return recommendations

        except Exception as e:
            st.error(f"Recommendation error: {str(e)}")
            return []

    # -----------------------------
    # BUTTON
    # -----------------------------
    if st.button("Get Recommendations"):

        recs = recommend_items(selected_attraction, top_n)

        if recs:
            st.success("✨ Recommended Attractions:")
            for r in recs:
                st.write("👉", r)
