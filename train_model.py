###with grid search###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if __name__ == '__main__':
    # 1. Load dataset
    df = pd.read_csv("california_housing_train.csv")

    print("Shape:", df.shape)
    print(df.head())
    print("Missing Values:\n", df.isnull().sum())


    # 2. Missing value handle
    df = df.fillna(df.median(numeric_only=True))


    # 3. Feature Engineering
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + 1)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + 1)
    df["population_per_household"] = df["population"] / (df["households"] + 1)


    # 4. X and y
    X = df.drop("median_house_value", axis=1)
    Y = df["median_house_value"]


    # 5. Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=55
    )


    # 6. Base Random Forest model
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )
    param_grid={
        "n_estimators":[100,200],
        "max_depth":[8,9,10],
        "min_samples_split":[2,5],
        "min_samples_leaf":[1,3,5],
        "max_features":["sqrt",None]
    }
    grid_search=GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=2

    )

    grid_search.fit(X_train,Y_train)


    best_model=grid_search.best_estimator_

    print(f"Best params :{grid_search.best_params_}")

    print(f"Best r2 score :{grid_search.best_score_}")

    y_pred=best_model.predict(X_test)










    # 11. Evaluation
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)

    print("\nFinal Test Performance:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


    # 12. Train vs Test score
    train_pred = best_model.predict(X_train)

    print("\nOverfitting Check:")
    print("Train R2:", r2_score(Y_train, train_pred))
    print("Test R2:", r2)


    # 13. Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))

    plt.scatter(Y_test, y_pred, alpha=0.5, label="Predicted Points")

    min_val = min(Y_test.min(), y_pred.min())
    max_val = max(Y_test.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        label="Perfect Prediction Line"
    )

    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title("Random Forest + GridSearchCV: Actual vs Predicted")
    plt.legend()
    plt.show()


    # 14. Feature Importance
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)


    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance - Best Random Forest Model")
    plt.show()


    joblib.dump(best_model, "california_random_forest_model.pkl")
    print("Model saved successfully!")