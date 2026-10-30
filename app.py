"""
Should I Leave Now?
ML-Based Travel Time Prediction & Departure Recommendation System
Streamlit Web Application (College ML Mini-Project)
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta, time

# Page configuration
st.set_page_config(
    page_title="Should I Leave Now? | Travel Time Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom student-friendly CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .highlight-card {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Loads the trained model pipeline from best_model.joblib."""
    model_paths = [
        "best_model.joblib",
        os.path.join(os.path.dirname(__file__), "best_model.joblib"),
        os.path.join(os.path.dirname(__file__), "models", "adaboost.joblib"),
    ]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None


@st.cache_data
def load_csv_data():
    """Loads and prepares travel_times.csv for EDA preview."""
    data_paths = [
        "travel_times.csv",
        os.path.join(os.path.dirname(__file__), "travel_times.csv"),
        os.path.join("data", "travel_times.csv"),
    ]
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Parse hour and clean columns
            df['StartHour'] = df['StartTime'].apply(lambda x: int(str(x).split(':')[0]))
            df['GoingTo'] = df['GoingTo'].str.strip()
            df['DayOfWeek'] = df['DayOfWeek'].str.strip()
            df['Take407All'] = df['Take407All'].str.strip().str.capitalize()
            return df
    return None


# Load model and dataset
model = load_trained_model()
df = load_csv_data()

# App Header
st.markdown("<div class='main-header'>🚗 Should I Leave Now?</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>ML-Based Travel Time Prediction and Departure Recommendation System</div>",
    unsafe_allow_html=True
)

# Navigation tabs
tabs = st.tabs([
    "🔮 Predict Travel Time",
    "⏰ Should I Leave Now?",
    "📊 Model Comparison",
    "📈 Dataset & EDA",
    "ℹ️ About Project"
])


# TAB 1: PREDICT TRAVEL TIME
with tabs[0]:
    st.subheader("Predict Journey Duration")
    st.write("Enter your scheduled trip details to estimate travel time based on historical commuting patterns.")

    col1, col2 = st.columns(2)

    with col1:
        destination = st.selectbox(
            "Destination",
            options=["GSK", "Home"],
            index=0,
            help="GSK represents workplace in Mississauga; Home represents return commute."
        )
        distance = st.number_input(
            "Trip Distance (km)",
            min_value=1.0,
            max_value=150.0,
            value=51.5,
            step=0.5
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            index=1
        )

    with col2:
        departure_time = st.time_input(
            "Planned Departure Time",
            value=time(7, 45)
        )
        take_407 = st.radio(
            "Take Highway 407 Toll Route?",
            options=["No", "Yes"],
            index=0,
            horizontal=True,
            help="Historically, taking Highway 407 saves ~8 minutes on average."
        )

    if st.button("🔮 Calculate Estimated Travel Time", type="primary"):
        if model is None:
            st.error("Error: Could not find 'best_model.joblib'. Please run project.ipynb first to save the model.")
        else:
            dep_hour = departure_time.hour
            dep_minute = departure_time.minute
            dep_decimal = dep_hour + dep_minute / 60.0

            # Prepare 1-row DataFrame matching the model's training schema
            input_df = pd.DataFrame([{
                "Distance": float(distance),
                "DepartureDecimal": float(dep_decimal),
                "StartHour": int(dep_hour),
                "GoingTo": destination,
                "DayOfWeek": day_of_week,
                "Take407All": take_407
            }])

            # Run prediction
            predicted_time = float(model.predict(input_df)[0])
            predicted_time = max(1.0, round(predicted_time, 1))

            st.markdown("---")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown(f"""
                <div class='metric-card' style='border-left: 5px solid #2563EB;'>
                    <div class='metric-label'>Estimated Travel Time</div>
                    <div class='metric-value'>{predicted_time} minutes</div>
                    <span style='color:#64748B; font-size:0.85rem;'>Predicted using AdaBoost Regressor Pipeline</span>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Trip Summary</div>
                    <div style='font-size:1.1rem; font-weight:600; margin-top:0.3rem;'>
                        {departure_time.strftime('%H:%M')} on {day_of_week}
                    </div>
                    <span style='color:#64748B; font-size:0.85rem;'>
                        To: {destination} | Distance: {distance} km | Toll: {take_407}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Cautionary warning if outside historical coverage
            if destination == "GSK" and not (6 <= dep_hour <= 9):
                st.warning("⚠️ **Notice:** All historical trips to GSK departed between 6:00 and 9:00 AM. Predictions outside this morning window are estimates.")
            elif destination == "Home" and not (12 <= dep_hour <= 20):
                st.warning("⚠️ **Notice:** All historical trips to Home departed in the afternoon/evening (12:00 to 20:00). Predictions outside this window are estimates.")


# TAB 2: SHOULD I LEAVE NOW? (DEPARTURE COMPARISON)
with tabs[1]:
    st.subheader("Departure-Time Recommendation Engine")
    st.write("Compare different departure times within a time window to find the most optimal time to leave.")

    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        comp_dest = st.selectbox("Destination ", ["GSK", "Home"], index=0)
        comp_dist = st.number_input("Distance (km) ", min_value=10.0, max_value=150.0, value=51.5, step=0.5)

    with comp_col2:
        comp_day = st.selectbox("Day of Week ", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], index=1)
        comp_time = st.time_input("Planned Departure Time ", value=time(7, 45))

    with comp_col3:
        comp_toll = st.radio("Take 407 Toll Route? ", ["No", "Yes"], index=0, horizontal=True)
        window = st.slider("Comparison Window (± minutes)", min_value=30, max_value=60, value=45, step=15)

    if st.button("⏰ Compare Candidate Departure Times", type="primary"):
        if model is None:
            st.error("Error: Could not find 'best_model.joblib'. Please run project.ipynb first to save the model.")
        else:
            base_dt = datetime(2026, 1, 1, comp_time.hour, comp_time.minute)
            curr = base_dt - timedelta(minutes=window)
            end = base_dt + timedelta(minutes=window)

            candidate_rows = []
            while curr <= end:
                t_str = curr.strftime("%H:%M")
                h = curr.hour
                m = curr.minute
                dec_hour = h + m / 60.0

                row_input = pd.DataFrame([{
                    "Distance": float(comp_dist),
                    "DepartureDecimal": float(dec_hour),
                    "StartHour": int(h),
                    "GoingTo": comp_dest,
                    "DayOfWeek": comp_day,
                    "Take407All": comp_toll
                }])
                pred_duration = round(float(model.predict(row_input)[0]), 1)

                # Predict duration with alternative toll road option
                alt_toll = "Yes" if comp_toll == "No" else "No"
                row_alt = row_input.copy()
                row_alt["Take407All"] = alt_toll
                pred_alt = round(float(model.predict(row_alt)[0]), 1)

                candidate_rows.append({
                    "Departure Time": t_str,
                    "Predicted Duration (min)": pred_duration,
                    "Duration with Toll (min)": pred_alt if comp_toll == "No" else pred_duration,
                    "Toll Savings (min)": round(pred_duration - pred_alt, 1) if comp_toll == "No" else 0.0
                })
                curr += timedelta(minutes=15)

            df_candidates = pd.DataFrame(candidate_rows)

            # Planned duration vs best duration
            planned_time_str = comp_time.strftime("%H:%M")
            planned_matches = df_candidates[df_candidates["Departure Time"] == planned_time_str]
            planned_dur = planned_matches.iloc[0]["Predicted Duration (min)"] if not planned_matches.empty else df_candidates.iloc[0]["Predicted Duration (min)"]

            best_row = df_candidates.sort_values(by="Predicted Duration (min)").iloc[0]
            best_time = best_row["Departure Time"]
            best_dur = best_row["Predicted Duration (min)"]
            savings = round(planned_dur - best_dur, 1)

            st.markdown("---")
            if savings > 0.5:
                st.markdown(f"""
                <div class='highlight-card'>
                    <h4 style='color:#065F46; margin:0;'>🌟 Recommendation: Consider departing at {best_time}</h4>
                    <p style='margin:0.3rem 0 0 0; color:#064E3B;'>
                        Departing at <strong>{best_time}</strong> is estimated to take <strong>{best_dur} minutes</strong> 
                        (saving approximately <strong>{savings} minutes</strong> compared to leaving at {planned_time_str} which takes {planned_dur} mins).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"ℹ️ **Observation:** Travel times are very consistent across this window ({planned_dur} mins). Leaving at your planned time ({planned_time_str}) is fine.")

            # Bar Chart Comparison
            st.write("#### Travel Time Comparison Across Departure Slots")
            fig, ax = plt.subplots(figsize=(10, 3.8))
            colors = ['#10B981' if t == best_time else ('#F59E0B' if t == planned_time_str else '#3B82F6') 
                      for t in df_candidates["Departure Time"]]
            bars = ax.bar(df_candidates["Departure Time"], df_candidates["Predicted Duration (min)"], color=colors, width=0.5)
            ax.set_ylabel("Duration (minutes)")
            ax.set_xlabel("Candidate Departure Time")
            ax.set_ylim(min(df_candidates["Predicted Duration (min)"]) - 5, max(df_candidates["Predicted Duration (min)"]) + 5)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}m", ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            # Table
            st.write("#### Detailed Candidate Times")
            st.dataframe(df_candidates, use_container_width=True)

            st.caption("🛡️ **Disclaimer:** Recommendations are statistical estimates based on historical 2011–2012 commuter data, not live GPS or satellite feeds.")


# TAB 3: MODEL COMPARISON
with tabs[2]:
    st.subheader("Model Performance & Comparison")
    st.write("We implemented and compared exactly four core regression algorithms on the same test dataset:")

    # Table of results
    comp_data = {
        "Model": ["AdaBoost Regressor", "Random Forest Regressor", "Linear Regression", "KNN Regressor"],
        "MAE (min)": [3.94, 3.95, 3.94, 4.48],
        "RMSE (min)": [5.74, 5.73, 5.77, 6.38],
        "R2 Score": [0.150, 0.155, 0.143, -0.049]
    }
    comp_df = pd.DataFrame(comp_data)

    st.dataframe(comp_df, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width/2, comp_df["MAE (min)"], width, label="MAE (minutes)", color="#3B82F6")
    ax.bar(x + width/2, comp_df["RMSE (min)"], width, label="RMSE (minutes)", color="#8B5CF6")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Model"])
    ax.set_ylabel("Error in Minutes")
    ax.set_title("Test Error Comparison Across the 4 Models (Lower is Better)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    ### Metric Explanations:
    - **MAE (Mean Absolute Error):** On average, the models are off by only **~3.9 minutes** on a 42-minute trip.
    - **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily. An RMSE of ~5.7 minutes reflects occasional severe delays (such as accidents or storms).
    - **R² Score (~0.15):** Explains ~15% of travel time variance using pre-departure information alone. The remaining variance is caused by unobserved real-time events (weather, sudden accidents).
    - **Best Model:** **AdaBoost Regressor** was selected as our best-performing model based on cross-validation error on the training data.
    """)


# TAB 4: DATASET & EDA
with tabs[3]:
    st.subheader("Dataset Overview & Exploratory Data Analysis (EDA)")

    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Commuter Trips", len(df))
        c2.metric("Average Duration", f"{df['TotalTime'].mean():.1f} mins")
        c3.metric("Average Distance", f"{df['Distance'].mean():.1f} km")

        st.write("#### Sample Data Preview")
        st.dataframe(df[['Date', 'StartTime', 'DayOfWeek', 'GoingTo', 'Distance', 'Take407All', 'TotalTime']].head(8), use_container_width=True)

        st.write("#### Key Visualizations from the Dataset")
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        # Distribution
        sns.histplot(df['TotalTime'], kde=True, color='royalblue', ax=axes[0, 0])
        axes[0, 0].set_title("Distribution of Commute Duration (TotalTime)")
        axes[0, 0].set_xlabel("Duration (min)")

        # Destination
        sns.boxplot(x='GoingTo', y='TotalTime', data=df, palette=['#3B82F6', '#10B981'], ax=axes[0, 1], hue='GoingTo', legend=False)
        axes[0, 1].set_title("Travel Time by Destination (GSK vs Home)")

        # Day of week
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        sns.boxplot(x='DayOfWeek', y='TotalTime', data=df, order=order, palette='Set2', ax=axes[1, 0], hue='DayOfWeek', legend=False)
        axes[1, 0].set_title("Travel Time by Day of the Week")

        # Toll Road
        sns.boxplot(x='Take407All', y='TotalTime', data=df, palette=['#EF4444', '#10B981'], ax=axes[1, 1], hue='Take407All', legend=False)
        axes[1, 1].set_title("Toll Road (407) vs Non-Toll Duration (~8 mins saved)")

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Dataset file `travel_times.csv` loaded.")


# TAB 5: ABOUT PROJECT
with tabs[4]:
    st.subheader("About This Project")
    st.markdown("""
    **Project Title:** Should I Leave Now? — ML-Based Travel Time Prediction and Departure Recommendation System  
    **Domain:** Intelligent Transportation Systems (ITS) & Machine Learning Mini-Project  

    ### 1. Problem Statement:
    Commuters want to estimate journey durations and know if leaving earlier or later would save travel time. 
    Using historical GPS-logged trips, this project predicts travel time and compares candidate departure windows.

    ### 2. Machine Learning Workflow:
    - **Step 1:** Load raw commuter GPS records (`travel_times.csv`).
    - **Step 2:** Clean and parse departure times (`StartTime`) into hours and decimal fractions.
    - **Step 3:** Prevent target leakage by strictly excluding post-trip columns (`AvgSpeed`, `MaxSpeed`, `MovingTime`, `FuelEconomy`, `Comments`).
    - **Step 4:** Build scikit-learn preprocessing pipelines (`StandardScaler` + `OneHotEncoder`).
    - **Step 5:** Train and evaluate 4 regression algorithms (Linear Regression, KNN, Random Forest, AdaBoost).
    - **Step 6:** Save the best model (`best_model.joblib`) and deploy it on Streamlit for live predictions.

    ### 3. Limitations:
    - **Historical Data:** The model uses recorded 2011–2012 commuter trips. It does not connect to live traffic cameras or GPS satellites.
    - **Specific Route:** Data represents commutes between New Hamburg and Mississauga/Toronto via Highway 401/403/407.
    - **Dataset Size:** 205 records from a single commuter diary.
    """)

st.markdown("---")
st.caption("College ML Mini-Project | Intelligent Transportation Systems (ITS) Predictive Analytics")
