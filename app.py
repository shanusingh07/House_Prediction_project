import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f4e79;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f5f7fa;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.metric-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
    border-left: 6px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("california_random_forest_model.pkl")

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🏠 California House Price Prediction</p>', unsafe_allow_html=True)
st.write("An interactive Machine Learning app using **Random Forest Regressor + GridSearchCV**.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Input Panel")
st.sidebar.write("Enter housing details below:")

longitude = st.sidebar.slider("Longitude", -125.0, -113.0, -122.23)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.88)
housing_median_age = st.sidebar.slider("Housing Median Age", 1.0, 60.0, 41.0)
total_rooms = st.sidebar.number_input("Total Rooms", min_value=1.0, value=880.0)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1.0, value=129.0)
population = st.sidebar.number_input("Population", min_value=1.0, value=322.0)
households = st.sidebar.number_input("Households", min_value=1.0, value=126.0)
median_income = st.sidebar.slider("Median Income", 0.5, 15.0, 8.3252)

# ---------------- FEATURE ENGINEERING ----------------
rooms_per_household = total_rooms / (households + 1)
bedrooms_per_room = total_bedrooms / (total_rooms + 1)
population_per_household = population / (households + 1)

input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_household": population_per_household
}])

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 Feature Importance",
    "📋 Input Data",
    "📘 About Project"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("House Price Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Median Income", f"{median_income}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Rooms / Household", f"{rooms_per_household:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Population / Household", f"{population_per_household:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    if st.button("🚀 Predict House Price", use_container_width=True):
        prediction = model.predict(input_data)[0]

        if prediction < 150000:
            category = "Low Price House"
        elif prediction < 300000:
            category = "Medium Price House"
        else:
            category = "High Price House"

        st.success("Prediction completed successfully!")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Predicted Price USD", f"${prediction:,.2f}")

        with c2:
            st.metric("Approx Price INR", f"₹{prediction * 83:,.2f}")

        with c3:
            st.metric("Category", category)

        report = input_data.copy()
        report["Predicted Price USD"] = prediction
        report["Predicted Price INR"] = prediction * 83
        report["Category"] = category

        csv = report.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Model Feature Importance")

    importance = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance, use_container_width=True)
    st.bar_chart(importance.set_index("Feature"))

    st.info("Higher importance means the feature has more effect on prediction.")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Input Data Preview")
    st.dataframe(input_data, use_container_width=True)

    st.subheader("Engineered Features")
    st.write(f"Rooms per Household: **{rooms_per_household:.2f}**")
    st.write(f"Bedrooms per Room: **{bedrooms_per_room:.4f}**")
    st.write(f"Population per Household: **{population_per_household:.2f}**")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("About This Project")

    st.markdown("""
    This project predicts California house prices using a Machine Learning model.

    ### Machine Learning Steps:
    - Data loading
    - Data cleaning
    - Feature engineering
    - Train-test split
    - Random Forest Regressor
    - GridSearchCV hyperparameter tuning
    - Model evaluation
    - Streamlit web app

    ### Technologies Used:
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Joblib
    """)

    st.success("This project is suitable for resume, GitHub, and portfolio.")