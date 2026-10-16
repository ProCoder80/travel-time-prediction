# Should I Leave Now? 🚗
### ML-Based Travel Time Prediction & Departure Recommendation System

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20scikit--learn-orange.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-22%20passed%20%7C%20100%25-brightgreen.svg)]()
[![Domain](https://img.shields.io/badge/domain-Intelligent%20Transportation%20Systems%20(ITS)-purple.svg)]()

---

## 📌 Project Overview
**“Should I Leave Now?”** is an end-to-end Machine Learning mini-project in the domain of **Intelligent Transportation Systems (ITS)**. Urban commuters frequently face unpredictable journey durations and uncertainty about when to depart. This application leverages empirical patterns learned from historical commuter GPS logs to:
1. Predict expected travel duration (in minutes) based **strictly on pre-departure information** (zero target leakage).
2. Evaluate candidate departure times within a user-defined window ($\pm 60$ min) to recommend potentially faster departure slots.
3. Provide transparent methodological safeguards that detect out-of-distribution queries and explain prediction bounds.

> **Important Scientific Scope:** This project relies on historical commuter trip records from the Ontario Highway 401/403/407 corridor. It does **not** claim live GPS satellite tracking, real-time traffic sensor integration, or guaranteed arrival times. Recommendations are statistical estimates based on empirical patterns.

---

## 🎯 Objectives
- **Zero-Leakage Feature Pipeline:** Process GPS logging data without leaking concurrent or post-trip metrics (`AvgSpeed`, `MovingTime`, `FuelEconomy`, etc.).
- **Model Comparison:** Train and evaluate exactly four core regression algorithms:
  - **Linear Regression** (Baseline)
  - **K-Nearest Neighbors (KNN)**
  - **Random Forest Regressor**
  - **AdaBoost Regressor**
- **Evaluation:** Measure performance using MAE, RMSE, and $R^2$ across 5-fold cross-validation and held-out test splits.
- **Departure Recommendation Engine:** Automatically evaluate candidate times at 15-minute intervals, calculate toll vs. non-toll trade-offs, and audit data coverage.
- **Interactive UI:** Deliver a responsive, modern Streamlit dashboard for interactive predictions, candidate comparisons, and exploratory data analysis.

---

## 📊 Dataset Description & Source
The project utilizes `travel_times.csv`, a real-world dataset recorded by a commuter in Ontario, Canada:
- **Total Records:** 205 commuter trips between July 11, 2011 and January 6, 2012.
- **Commute Corridor:** Trips between New Hamburg and Mississauga/Toronto via Highway 401, 403, and Highway 407 ETR toll highway.
- **Key Attributes:**
  - `Date`: Commute date (multi-format parsed with DayOfWeek validation).
  - `StartTime`: Departure time (morning rush for GSK; afternoon/evening for Home).
  - `DayOfWeek`: Monday through Friday (workday commutes).
  - `GoingTo`: Workplace (`GSK`, 105 trips) or residence (`Home`, 100 trips).
  - `Distance`: Trip distance in km ($\mu = 50.98$ km, range $48.32$ to $60.32$ km).
  - `Take407All`: Toll usage (`No`: 170 trips, `Yes`: 35 trips). Taking Highway 407 saves $\approx 8.2$ minutes on average.
  - `TotalTime`: **Target variable** ($\mu = 41.90$ min, median $41.30$ min, range $28.2$ to $82.3$ min).

---

## 🏗️ Project Folder Structure
```text
travel-time-project/
├── app.py                     # Streamlit web application dashboard
├── requirements.txt           # Verified package dependencies
├── README.md                  # Comprehensive documentation and guide
├── .gitignore                 # Version control ignore rules
├── pytest.ini                 # Pytest configuration and warning filters
├── travel_times.csv           # Original dataset
├── data/
│   ├── .gitkeep
│   └── travel_times.csv       # Mirrored dataset copy
├── models/
│   ├── .gitkeep
│   ├── linear_regression.joblib
│   ├── knn_regressor.joblib
│   ├── random_forest.joblib
│   ├── adaboost.joblib
│   ├── training_metadata.joblib
│   └── test_eval_bundle.joblib
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Loading, date/time parsing, validation, splitting
│   ├── feature_engineering.py # Transformers, preprocessor pipeline, safeguards
│   ├── train_models.py        # 4-model training, 5-fold CV, model persistence
│   ├── evaluate_models.py     # Metrics (MAE, RMSE, R2) and diagnostic charts
│   ├── recommendation.py      # Candidate departure evaluation & ranking engine
│   └── predict.py             # Single-trip inference interface
├── tests/
│   ├── test_preprocessing.py  # Preprocessing, date parsing, leakage tests
│   ├── test_prediction.py     # Model inference, boundary, error tests
│   └── test_recommendation.py # Recommendation, ranking, coverage tests
├── notebooks/
│   └── eda.py                 # Standalone reproducible EDA script
├── reports/
│   ├── model_comparison.csv   # Empirical metrics across all 4 models
│   ├── project_report.md      # Full 15-section academic project report
│   └── figures/               # High-res diagnostic & EDA charts
│       ├── actual_vs_predicted.png
│       ├── eda_correlation_heatmap.png
│       ├── eda_dayofweek_vs_time.png
│       ├── eda_destination_comparison.png
│       ├── eda_distance_vs_time.png
│       ├── eda_hour_vs_time.png
│       ├── eda_totaltime_dist.png
│       ├── model_comparison.png
│       └── residual_analysis.png
└── screenshots/
    └── .gitkeep
```

---

## 📈 Empirical Model Comparison Results

All four models were trained on an 80/20 stratified split ($N_{\text{train}}=164, N_{\text{test}}=41$) with 5-fold cross-validation:

| Model | CV MAE (min) | CV RMSE (min) | CV $R^2$ | Test MAE (min) | Test RMSE (min) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost Regressor (Best)** | **3.90** | **5.97** | **0.138** | **3.94** | **5.74** | **0.150** |
| **Random Forest Regressor** | 3.96 | 6.09 | 0.093 | 3.95 | 5.73 | 0.155 |
| **Linear Regression (Baseline)** | 4.26 | 5.89 | 0.156 | 3.94 | 5.77 | 0.143 |
| **KNN Regressor ($k=7$)** | 4.27 | 5.83 | 0.200 | 4.48 | 6.38 | -0.049 |

### Key Findings
1. **Low Relative Error:** Mean Absolute Error is under **4 minutes** for an average 42-minute trip (~9.3% error).
2. **Variance Explanation:** Pre-departure features account for ~15% of total variance ($R^2 \approx 0.15$). Unobserved factors (unannounced lane closures, precipitation, accidents) account for remaining variance.
3. **Best Model Selection:** **AdaBoost Regressor** achieved the lowest cross-validation MAE (3.90 min) and was designated as the default production model.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.11, 3.12, or 3.14 is installed on your system.

### 2. Windows PowerShell Setup
```powershell
# Open Windows PowerShell and navigate to the project directory
cd c:\Users\HP\OneDrive\Desktop\Kartikeya\ML_PROJECT

# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Linux / macOS Terminal Setup
```bash
cd /path/to/ML_PROJECT
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Execution Commands

### 1. Train and Benchmark the Models
Trains all 4 models, performs 5-fold cross-validation, saves artifacts to `models/`, and generates `reports/model_comparison.csv`:
```powershell
python -m src.train_models
```

### 2. Generate EDA Figures
Generates all 6 publication-quality exploratory figures in `reports/figures/`:
```powershell
python -m notebooks.eda
```

### 3. Run Automated Tests
Executes the comprehensive 22-test suite covering preprocessing, leakage prevention, predictions, and recommendations:
```powershell
python -m pytest -v
```

### 4. Launch the Streamlit Dashboard
Launches the interactive web application on your local browser:
```powershell
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to interact with the application.

---

## 🧪 Automated Testing Suite
The project includes 22 rigorous automated test cases:
- `test_preprocessing.py`: CSV loading, missing values, multi-format date parsing, time extraction, target leakage prevention, stratified & chronological splits.
- `test_prediction.py`: Model availability, positive duration bounds, invalid destination rejection, negative/extreme distance rejection, malformed time strings, missing file handling, coverage auditing.
- `test_recommendation.py`: Candidate time interval generation, duration delta calculations, ranking order, toll alternative comparisons, unsupported query detection.

**Test Execution Result:** `22 passed in 4.00s (100% pass rate)`.

---

## 🌐 Publishing to GitHub

To publish this project to GitHub, execute the following commands in Windows PowerShell:

```powershell
# 1. Initialize git repository
git init

# 2. Stage all project files
git add .

# 3. Create initial commit
git commit -m "feat: complete 'Should I Leave Now?' travel time prediction and departure recommendation system"

# 4. Link to your remote GitHub repository
# (Create a new empty repository on GitHub first, then replace <your-repo-url>)
git remote add origin https://github.com/<your-username>/should-i-leave-now.git

# 5. Set branch to main and push
git branch -M main
git push -u origin main
```

---

## ⚠️ Known Limitations & Future Scope

### Limitations
- **Historical Data Only:** The dataset reflects commute conditions from 2011–2012; it does not connect to live traffic sensors.
- **Corridor-Specific:** Trained specifically for the New Hamburg to Mississauga corridor (Ontario 401/403/407).
- **Small Sample Size:** 205 trips limit deep learning or extensive hyperparameter exploration.

### Future Scope
- Integration with live open transit APIs (e.g., Ontario 511 Open Data).
- Weather enrichment (real-time rain, snow, temperature).
- Conformal prediction intervals for 90% confidence duration bounds.

---

## 📜 License & Acknowledgments
- Dataset: Kaggle Open Datasets (`travel_times.csv`).
- Course: College Machine Learning Mini-Project.
