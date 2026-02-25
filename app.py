import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

# =============================
# LOAD ARTIFACTS (SAME AS YOUR WORKING VERSION)
# =============================
@st.cache_resource
def load_artifacts():
    reg_pipeline = joblib.load("artifacts/regression_pipeline.pkl")
    clf_pipeline = joblib.load("artifacts/classification_pipeline.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")

    # recommender artifacts
    rec_df = joblib.load("artifacts/rec_df.pkl")
    item_features = joblib.load("artifacts/item_features.pkl")
    item_id_to_pos = joblib.load("artifacts/item_id_to_pos.pkl")

    return reg_pipeline, clf_pipeline, label_encoder, rec_df, item_features, item_id_to_pos


reg_pipeline, clf_pipeline, label_encoder, rec_df, item_features, item_id_to_pos = load_artifacts()

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("🔎 Choose Function")

page = st.sidebar.selectbox(
    "Navigation",
    ["⭐ Rating Prediction", "🎯 Visit Mode Classification", "🤖 Attraction Recommender"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")

# ============================================================
# PAGE 1 — REGRESSION
# ============================================================
if page == "⭐ Rating Prediction":

    st.title("🌍 Tourism Experience Analytics")
    st.subheader("Predict Expected Rating")

    col1, col2, col3 = st.columns(3)

    with col1:
        visit_year = st.number_input("Visit Year", 2015, 2030, 2024)
        visit_month = st.number_input("Visit Month", 1, 12, 6)

    with col2:
        country = st.selectbox("Country", sorted(rec_df["Country"].dropna().unique()))
        region = st.selectbox("Region", sorted(rec_df["Region"].dropna().unique()))

    with col3:
        attraction_type = st.selectbox(
            "Attraction Type",
            sorted(rec_df["AttractionType"].dropna().unique())
        )

    if st.button("Predict Rating"):

        input_df = pd.DataFrame([{
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "Country": country,
            "Region": region,
            "AttractionType": attraction_type
        }])

        pred_rating = reg_pipeline.predict(input_df)[0]

        st.success(f"⭐ Predicted Rating: {pred_rating:.2f}")

# ============================================================
# PAGE 2 — CLASSIFICATION
# ============================================================
elif page == "🎯 Visit Mode Classification":

    st.title("🎯 Visit Mode Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        visit_year = st.number_input("Visit Year", 2015, 2030, 2024, key="clf_year")
        visit_month = st.number_input("Visit Month", 1, 12, 6, key="clf_month")

    with col2:
        country = st.selectbox(
            "Country",
            sorted(rec_df["Country"].dropna().unique()),
            key="clf_country"
        )
        region = st.selectbox(
            "Region",
            sorted(rec_df["Region"].dropna().unique()),
            key="clf_region"
        )

    with col3:
        attraction_type = st.selectbox(
            "Attraction Type",
            sorted(rec_df["AttractionType"].dropna().unique()),
            key="clf_type"
        )

    if st.button("Predict Visit Mode"):

        input_df = pd.DataFrame([{
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "Country": country,
            "Region": region,
            "AttractionType": attraction_type
        }])

        pred_class = clf_pipeline.predict(input_df)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        st.info(f"🎯 Predicted Visit Mode: **{pred_label}**")

# ============================================================
# PAGE 3 — RECOMMENDER
# ============================================================
else:

    st.title("🤖 Attraction Recommender")

    attraction_list = sorted(rec_df["Attraction"].dropna().unique())
    selected_attr = st.selectbox("Select an attraction", attraction_list)

    def recommend_items(attraction_name, top_n=5):
        idx_list = rec_df.index[rec_df["Attraction"] == attraction_name].tolist()
        if not idx_list:
            return []

        idx = idx_list[0]
        pos = item_id_to_pos.get(idx)

        if pos is None:
            return []

        sim_scores = item_features[pos].dot(item_features.T)
        top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]

        return rec_df.iloc[top_indices]["Attraction"].values

    if st.button("Recommend"):

        recs = recommend_items(selected_attr, top_n=5)

        if len(recs) == 0:
            st.warning("No recommendations found.")
        else:
            st.success("✨ Recommended Attractions:")
            for r in recs:
                st.write("•", r)
