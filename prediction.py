import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


def train_model(df):

    features = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "CGPA"
    ]

    target = "Chance of Admit"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n========== MODEL EVALUATION ==========")

    print("R2 Score:", round(r2_score(y_test, y_pred), 4))

    print("Mean Squared Error:",
          round(mean_squared_error(y_test, y_pred), 4))

    print("Mean Absolute Error:",
          round(mean_absolute_error(y_test, y_pred), 4))

    print("\nModel Training Completed Successfully!")

    return model


def predict_admission(model, gre, toefl, rating, cgpa):

    sample = pd.DataFrame([{
        "GRE Score": gre,
        "TOEFL Score": toefl,
        "University Rating": rating,
        "CGPA": cgpa
    }])

    prediction = model.predict(sample)[0]

    print("\n========== ADMISSION PREDICTION ==========")

    print("GRE Score:", gre)
    print("TOEFL Score:", toefl)
    print("University Rating:", rating)
    print("CGPA:", cgpa)

    print("Predicted Chance of Admit:",
          round(prediction, 4))

    print("Predicted Admission Percentage:",
          round(prediction * 100, 2), "%")
