# House_Prediction_project
# 🏠 California House Price Prediction (ML + Streamlit App)

An interactive Machine Learning web application that predicts California house prices using a **Random Forest Regressor** with **GridSearchCV optimization**.

---

## 🚀 Live Project Features

* 🔮 Real-time house price prediction
* 🎛️ Interactive sidebar inputs (sliders + inputs)
* 📊 Feature importance visualization
* 📈 Engineered features display
* 💾 Download prediction report (CSV)
* 💡 Price category (Low / Medium / High)
* 🇮🇳 INR conversion included
* 🧠 Clean UI with tabs & metrics

---

## 🧠 Machine Learning Pipeline

1. Data Loading
2. Data Cleaning (Missing values handled)
3. Feature Engineering
4. Train-Test Split
5. Model Training (Random Forest)
6. Hyperparameter Tuning (GridSearchCV)
7. Model Evaluation
8. Model Deployment using Streamlit

---

## 📊 Features Used

### Original Features:

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income

### Engineered Features:

* Rooms per Household
* Bedrooms per Room
* Population per Household

---

## 🧮 Model Performance

> (Fill your actual values here 👇)

* **R2 Score:** 0.XX
* **RMSE:** XXXXX
* **MAE:** XXXXX

---

## 📁 Project Structure

```text
House_Prediction_project/
│
├── app.py                               # Streamlit UI
├── train_model.py                      # Model training script
├── california_random_forest_model.pkl  # Saved model
├── california_housing_train.csv        # Dataset
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Streamlit
* Joblib

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/shanusingh07/House_Prediction_project.git
cd House_Prediction_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run App

```bash
streamlit run app.py
```

---

## 📸 App UI Preview

> (Add screenshot here for better impact)

---

## 📊 Output Example

* Predicted House Price (USD 💲)
* Approx Price in INR 🇮🇳
* Price Category (Low / Medium / High)
* Downloadable CSV Report

---

## 📥 Download Feature

Users can download prediction results with:

* Input values
* Predicted price
* Category

---

## 🌟 Future Improvements

* Add advanced models (XGBoost, Gradient Boosting)
* Add map visualization (location-based prediction)
* Deploy on Streamlit Cloud / Render
* Improve UI animations
* Add user login system

---

## 🙌 Author

**Shanu Singh**
🔗 GitHub: https://github.com/shanusingh07

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

## 📌 Note

This project is built for learning, portfolio, and demonstration purposes.
