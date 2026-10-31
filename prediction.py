import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_model(df):
    required = ["admit", "gre", "gpa", "prestige"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["admit"])
    X = df[["gre", "gpa", "prestige"]]
    y = df["admit"]

    valid = X.notna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Need at least two admission classes with at least "
            "two records in each class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nMODEL EVALUATION")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model


def predict_admission(model, gre, gpa, prestige):
    sample = pd.DataFrame([{
        "gre": gre,
        "gpa": gpa,
        "prestige": prestige
    }])

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = list(model.named_steps["logisticregression"].classes_)

    print("\nADMISSION PREDICTION")
    print("Prediction:", "Likely to be admitted" if prediction == 1
          else "Less likely to be admitted")

    if 1 in classes:
        admit_index = classes.index(1)
        print(
            f"Estimated admission probability: "
            f"{probabilities[admit_index]:.2%}"
        )
