import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("california_random_forest_model.pkl")

# -------------------- TITLE --------------------
st.title("🏠 California House Price Prediction")
st.write("This app predicts California house prices using a Random Forest Regression model.")

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("Enter House Details")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.number_input("Housing Median Age", value=41.0, min_value=0.0)
total_rooms = st.sidebar.number_input("Total Rooms", value=880.0, min_value=0.0)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=129.0, min_value=0.0)
population = st.sidebar.number_input("Population", value=322.0, min_value=0.0)
households = st.sidebar.number_input("Households", value=126.0, min_value=0.0)
median_income = st.sidebar.number_input("Median Income", value=8.3252, min_value=0.0)

# -------------------- FEATURE ENGINEERING --------------------
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

# -------------------- MAIN LAYOUT --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Input Data Preview")
    st.dataframe(input_data)

with col2:
    st.subheader("ℹ️ Engineered Features")
    st.write(f"Rooms per Household: **{rooms_per_household:.2f}**")
    st.write(f"Bedrooms per Room: **{bedrooms_per_room:.2f}**")
    st.write(f"Population per Household: **{population_per_household:.2f}**")

# -------------------- PREDICTION --------------------
st.divider()

if st.button("🔮 Predict House Price"):
    if total_rooms <= 0 or households <= 0 or median_income <= 0:
        st.error("Please enter valid positive values for rooms, households, and income.")
    else:
        prediction = model.predict(input_data)[0]

        if prediction < 150000:
            category = "Low Price House"
        elif prediction < 300000:
            category = "Medium Price House"
        else:
            category = "High Price House"

        st.success(f"Predicted House Price: ${prediction:,.2f}")
        st.info(f"Approx Price in Indian Rupees: ₹{prediction * 83:,.2f}")
        st.write(f"House Category: **{category}**")

        report = input_data.copy()
        report["Predicted Price USD"] = prediction
        report["Predicted Price INR"] = prediction * 83
        report["Category"] = category

        csv = report.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="prediction_report.csv",
            mime="text/csv"
        )

# -------------------- FEATURE IMPORTANCE --------------------
st.divider()
st.subheader("📊 Model Feature Importance")

importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

# -------------------- ABOUT PROJECT --------------------
st.divider()
st.subheader("📘 About This Project")

st.markdown("""
This project predicts California house prices using a **Random Forest Regressor** model.

### Steps Used:
- Data loading
- Data cleaning
- Feature engineering
- Train-test split
- Random Forest model training
- GridSearchCV hyperparameter tuning
- Model evaluation
- Streamlit web app deployment

### Features Used:
- Longitude
- Latitude
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Median income
- Rooms per household
- Bedrooms per room
- Population per household
""")