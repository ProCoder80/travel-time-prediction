# Should I Leave Now? 🚗
### ML-Based Travel Time Prediction & Departure Recommendation System
**College Machine Learning Mini-Project / Lab Practical**

---

## 📌 Project Overview
**“Should I Leave Now?”** is an Intelligent Transportation System (ITS) machine learning mini-project built for a college ML lab course. 

Commuters regularly ask: *"Should I leave now, or will waiting 15–30 minutes result in a smoother, faster trip?"*

This project uses historical commuter GPS records (`travel_times.csv`) to:
1. Predict total journey duration (`TotalTime` in minutes) based **strictly on pre-departure information** (zero target leakage).
2. Compare candidate departure times within a selected time window ($\pm 45$ to $\pm 60$ minutes) to identify potentially faster departure slots.
3. Train, evaluate, and compare four regression algorithms:
   - **Linear Regression** (Baseline model)
   - **K-Nearest Neighbors (KNN)**
   - **Random Forest Regressor**
   - **AdaBoost Regressor**
4. Deploy an interactive web application using **Streamlit** for real-time predictions and departure-time comparisons.

---

## 📁 Simple Project Structure
```text
ML_PROJECT/
├── project.ipynb       # Complete ML pipeline in a single notebook (EDA, training, evaluation, comparison)
├── app.py              # Self-contained Streamlit web application
├── travel_times.csv    # Commuter dataset (205 trips)
├── best_model.joblib   # Saved trained model (AdaBoost Pipeline)
├── requirements.txt    # Basic project dependencies
└── README.md           # Project documentation and setup guide
```

---

## 🔬 How the Project Works

### 1. `project.ipynb` (Jupyter Notebook)
Contains the entire data science and machine learning workflow:
- **Step 1: Imports:** Standard libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `joblib`).
- **Step 2: Loading Data:** Loads `travel_times.csv` and inspects rows, columns, and data types.
- **Step 3: Data Cleaning & Preprocessing:** 
  - Extracts departure hour and decimal time from `StartTime`.
  - Strictly drops post-trip telemetry (`AvgSpeed`, `MaxSpeed`, `MovingTime`, `FuelEconomy`, `Comments`) to prevent **target leakage**.
- **Step 4: Exploratory Data Analysis (EDA):** Plots travel time distributions, destination comparisons (GSK vs Home), weekday trends, and toll road savings.
- **Step 5: Feature Engineering:** Encodes categories with `OneHotEncoder` and scales numerical columns with `StandardScaler` inside a scikit-learn `ColumnTransformer`.
- **Step 6: Train-Test Split:** 80/20 train/test split.
- **Step 7: Model Training:** Trains all 4 models (Linear Regression, KNN, Random Forest, AdaBoost).
- **Step 8: Model Evaluation & Comparison:** Calculates and compares MAE, RMSE, and $R^2$ score on the test set.
- **Step 9: Departure Recommendation Function:** Implements and tests candidate departure slot comparison.
- **Step 10: Saving Model:** Saves the top model to `best_model.joblib`.

### 2. `app.py` (Streamlit Web App)
A single Python script that loads `best_model.joblib` and provides:
- **Tab 1: 🔮 Predict Travel Time:** Form to enter destination, distance, day, departure time, and toll road choice to predict estimated journey duration.
- **Tab 2: ⏰ Should I Leave Now?:** Departure-time recommendation engine that compares candidate departure slots every 15 minutes across a user-defined window, highlights the fastest departure time, and displays toll road savings (~8 mins).
- **Tab 3: 📊 Model Comparison:** Displays comparison metrics (MAE, RMSE, $R^2$) and comparison bar charts across the four models.
- **Tab 4: 📈 Dataset & EDA:** Interactive summary metrics and exploratory data analysis charts.
- **Tab 5: ℹ️ About Project:** Problem statement, algorithm descriptions, and limitations.

---

## 📊 Model Comparison Results

| Model | MAE (min) | RMSE (min) | $R^2$ Score | Performance Summary |
| :--- | :---: | :---: | :---: | :--- |
| 🥇 **AdaBoost Regressor** | **3.94** | **5.74** | **0.150** | **Best overall model** (Lowest CV & Test MAE) |
| 🥈 **Random Forest Regressor** | 3.95 | 5.73 | 0.155 | High stability, low variance |
| 🥉 **Linear Regression** | 3.94 | 5.77 | 0.143 | Simple, interpretable baseline |
| 4️⃣ **KNN Regressor ($k=7$)** | 4.48 | 6.38 | -0.049 | Distance-weighted neighbors |

- **Average Commute Duration:** $\approx 42$ minutes.
- **Average Error:** Under **4 minutes** ($\approx 9.3\%$ relative error).
- **Toll Road Effect:** Taking Highway 407 saves $\approx \mathbf{8.2\text{ minutes}}$ on average.

---

## 🚀 How to Run the Project

### Option A: Run the Jupyter Notebook (`project.ipynb`)
1. Open the folder in VS Code or Jupyter Notebook:
   ```bash
   jupyter notebook project.ipynb
   ```
2. Click **"Run All"** to execute the cells step-by-step from data loading to model evaluation.
3. The notebook will automatically train the models, display comparison plots, and save `best_model.joblib`.

### Option B: Run the Streamlit Web Application (`app.py`)
1. Open your terminal / Windows PowerShell in the project directory.
2. Run:
   ```bash
   streamlit run app.py
   ```
3. The app will launch in your browser at **http://localhost:8501**.

---

## 📦 Requirements
Install all dependencies using:
```bash
pip install -r requirements.txt
```
Contents of `requirements.txt`:
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.30.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## ⚠️ Limitations
- **Historical Data:** The dataset consists of 205 recorded commuter trips along the Ontario 401/403/407 corridor (2011–2012). It does not use live satellite/GPS traffic feeds.
- **Pre-departure Prediction:** Predicts expected duration using known scheduling factors; unexpected real-time accidents or storms create natural variation ($\pm 4$ min).
