from analysis import load_data, analyze_data
from prediction import train_model, predict_admission


def main():

    data = load_data()

    # Dataset Analysis
    analyze_data(data)

    # Model Training and Evaluation
    model = train_model(data)

    # Sample Prediction
    print("\nEnter Student Details")

    gre = float(input("Enter GRE Score: "))

    toefl = float(input("Enter TOEFL Score: "))

    rating = int(input("Enter University Rating (1-5): "))

    cgpa = float(input("Enter CGPA: "))

    if not 1 <= rating <= 5:
        raise ValueError("University Rating must be between 1 and 5.")

    if not 0 <= cgpa <= 10:
        raise ValueError("CGPA must be between 0 and 10.")

    predict_admission(model, gre, toefl, rating, cgpa)


if __name__ == "__main__":
    main()
