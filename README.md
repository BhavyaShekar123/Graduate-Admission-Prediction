# Graduate Admission Prediction using Machine Learning

## Project features
- Load and inspect an admissions CSV dataset
- Display histograms for numeric columns
- Train and evaluate a Logistic Regression classifier
- Predict admission outcome from GRE, GPA, and university prestige

## Folder structure
```text
graduate_admission_prediction/
├── data/
│   └── admission.csv
├── src/
│   ├── main.py
│   ├── analysis.py
│   └── prediction.py
├── requirements.txt
└── README.md
```

## Dataset
Place your real `admission.csv` file in the `data` folder. It should contain these columns:
`admit`, `gre`, `gpa`, `prestige`.

The commonly used `binary.csv` admissions dataset has these columns; it can be renamed to `admission.csv`.

## Install and run (PowerShell)
From the project root folder:
```powershell
python -m pip install -r requirements.txt
python src/main.py
```

Predictions are estimates from the dataset, not guarantees of actual admission.
