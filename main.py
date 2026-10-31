from analysis import load_data, analyze_data
from prediction import train_model, predict_admission


def main():
    data = load_data()

    # Step 1: Explore the dataset and display histograms
    analyze_data(data)

    # Step 2: Train and evaluate the ML model
    model = train_model(data)

    # Step 3: Get input and predict admission
    print("\nEnter student details to predict admission.")
    gre = float(input("GRE score: "))
    gpa = float(input("GPA: "))
    prestige = int(input("University prestige (1-4): "))

    if not 1 <= prestige <= 4:
        raise ValueError("Prestige must be between 1 and 4.")

    predict_admission(model, gre, gpa, prestige)


if __name__ == "__main__":
    main()
