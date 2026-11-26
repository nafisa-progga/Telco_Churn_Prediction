# Telco Customer Churn Prediction

An end-to-end MLOps pipeline for predicting customer churn on the Telco Customer Churn dataset. It includes exploratory data analysis, modular data cleaning and feature engineering, candidate model training and selection, Population Stability Index (PSI) data drift monitoring with timestamped run logging, unit and API testing with `pytest`, CI via GitHub Actions, and real-time inference serving with FastAPI and Docker.

## Features

- **Data Processing & Feature Engineering (`src/process.py`):** Cleans raw telecom records, converts `TotalCharges` to numeric, removes identifier columns (`customerID`), maps `Churn` to binary `0`/`1`, engineers `tenure_group` and `avg_monthly_spend`, and encapsulates imputation, scaling, and one-hot encoding in a reusable `ColumnTransformer`.
- **Model Training & Selection (`src/train.py`):** Trains candidate classifiers (Logistic Regression and Random Forest) on a stratified train/test split, evaluates Accuracy, Precision, Recall, F1, and ROC-AUC, and exports the best model pipeline, metrics report, and feature schema to `artifacts/`.
- **Inference Engine (`src/predict.py`):** Aligns incoming customer payloads to the saved training schema and outputs churn probabilities, binary predictions, and labels (`Yes`/`No`).
- **Drift Monitoring (`src/monitor.py`):** Computes Population Stability Index (PSI) across numeric features against a reference dataset, saving JSON snapshots to `snapshots/` and timestamped run logs to `snapshots/logs/`.
- **REST API (`app/main.py`):** Serves `/health` and `/predict` endpoints with Pydantic input validation and configurable probability thresholds.
- **Automated Testing & CI (`tests/`, `.github/workflows/ci.yml`):** Validates preprocessing, drift calculations, and API endpoints on every push and pull request.

## Project Structure

```text
Telco_Churn_Prediction/
├── .github/workflows/ci.yml   # GitHub Actions CI workflow running pytest
├── app/
│   └── main.py                # FastAPI service (/health, /predict)
├── artifacts/                 # Trained model pipeline, metrics, and feature schema
├── data/
│   ├── processed/             # Cleaned and feature-engineered dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis notebook
├── snapshots/
│   └── logs/                  # Drift snapshots and timestamped monitoring logs
├── src/
│   ├── process.py             # Cleaning, feature engineering, and ColumnTransformer
│   ├── train.py               # Model training, evaluation, and artifact export
│   ├── predict.py             # Batch and single-customer churn inference
│   └── monitor.py             # PSI drift detection, snapshots, and run logs
├── tests/
│   ├── test_api.py            # FastAPI endpoint and schema validation tests
│   └── test_preprocess.py     # Preprocessing, feature engineering, and drift tests
├── .env                       # Local environment variables (gitignored)
├── .gitignore                 # Git ignore rules
├── Dockerfile                 # Container definition for FastAPI serving
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Setup & Usage

### 1. Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Create a `.env` file in the project root:

```env
APP_ENV=dev
CHURN_THRESHOLD=0.5
```

### 3. Run Data Processing, Model Training & Drift Monitoring

```bash
# Clean raw data, engineer features, and save processed CSV under data/processed/
python src/process.py

# Train candidate models and save best model, metrics, and schema under artifacts/
python src/train.py

# Run PSI drift checks and save drift snapshot and log under snapshots/
python src/monitor.py
```

### 4. Start the FastAPI Inference Server

```bash
uvicorn app.main:app --reload --port 8000
```

- **Health Check:** `GET http://localhost:8000/health`
- **Interactive API Docs:** `http://localhost:8000/docs`
- **Predict Churn:** `POST http://localhost:8000/predict`

#### Example `/predict` Request

```json
{
  "customers": [
    {
      "customerID": "0001-SAMPLE",
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 3,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 85.0,
      "TotalCharges": "255.0"
    }
  ],
  "threshold": 0.5
}
```

### 5. Run Tests & Docker

```bash
# Run unit and API test suite
pytest -v

# Build and run Docker container (ensure artifacts/ is generated first)
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```