"""
Should I Leave Now?
ML-Based Travel Time Prediction & Departure Recommendation System
Streamlit Dashboard Application
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time

from src.data_preprocessing import load_dataset, clean_and_prepare_dataset
from src.predict import (
    predict_travel_time,
    get_available_models,
    get_default_model_name,
)
from src.recommendation import DepartureRecommendationEngine
from src.evaluate_models import get_metric_explanations, calculate_metrics

# Page configuration
st.set_page_config(
    page_title="Should I Leave Now? | Travel Time Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, professional styling
st.markdown("""
<style>
    /* Global font & typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Metric Cards */
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        text-align: left;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-sub {
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    /* Banner Alert Boxes */
    .limitation-box {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    .recommendation-badge {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 4px solid #059669;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #065F46;
        margin-bottom: 1.2rem;
    }

    /* Header styling */
    .hero-title {
        color: #0F172A;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_dataset():
    """Safely loads and caches the prepared dataset for UI EDA."""
    raw = load_dataset()
    cleaned = clean_and_prepare_dataset(raw)
    return cleaned


@st.cache_data
def get_model_comparison_table():
    """Loads saved model evaluation metrics table."""
    path = os.path.join(os.path.dirname(__file__), "reports", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# Sidebar Navigation
st.sidebar.markdown("### 🚗 Should I Leave Now?")
st.sidebar.markdown(
    "<span style='font-size:0.85rem; color:#64748B;'>ML-Based Travel Time Prediction & Departure Guidance</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Home / Dashboard",
        "Predict Travel Time",
        "Departure-Time Comparison",
        "Model Performance",
        "Exploratory Data Analysis",
        "About & Methodology",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **System Status:**  
    🟢 4 Models Operational  
    📊 Dataset: 205 Commuter Trips  
    🏷️ Corridor: Ontario Hwy 401/403/407  
    """
)


# ==============================================================================
# PAGE 1: HOME / DASHBOARD
# ==============================================================================
if nav_choice == "Home / Dashboard":
    st.markdown("<div class='hero-title'>Should I Leave Now?</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Predicting journey duration and comparing departure windows using historical commuting patterns.</div>",
        unsafe_allow_html=True,
    )

    # Mandatory Limitations Notice
    st.markdown(
        """
        <div class='limitation-box'>
            <strong>Important Scientific Notice & Scope Limitation:</strong><br>
            This system is trained on historical commuter GPS trips (recorded between July 2011 and January 2012) 
            across the Highway 401 / 403 / 407 corridor. It does <strong>not</strong> use live GPS satellite tracking, 
            real-time traffic sensor feeds, or live congestion monitoring. All predictions and departure comparisons 
            are statistical estimates derived strictly from historical data patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key KPI metrics
    df = get_cached_dataset()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Total Trips</div>
                <div class='kpi-value'>{len(df)}</div>
                <div class='kpi-sub'>Mon–Fri Commutes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Avg Commute Duration</div>
                <div class='kpi-value'>{df['TotalTime'].mean():.1f} min</div>
                <div class='kpi-sub'>Median: {df['TotalTime'].median():.1f} min</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Avg Trip Distance</div>
                <div class='kpi-value'>{df['Distance'].mean():.1f} km</div>
                <div class='kpi-sub'>Range: {df['Distance'].min():.1f} – {df['Distance'].max():.1f} km</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        best_model = get_default_model_name()
        st.markdown(
            f"""
            <div class='kpi-card'>
                <div class='kpi-title'>Active Model</div>
                <div class='kpi-value' style='font-size:1.45rem;'>{best_model}</div>
                <div class='kpi-sub'>CV MAE: 3.90 min</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Workflow explanation
    st.markdown("<div class='section-title'>How The System Works</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            **1. Input Trip Details**  
            Provide scheduled destination, route distance, candidate departure time, day of week, and optional toll choice.
            """
        )
    with c2:
        st.markdown(
            """
            **2. Supervised Regression**  
            Pre-departure features are preprocessed (scaled & one-hot encoded) and passed through four scikit-learn regression pipelines.
            """
        )
    with c3:
        st.markdown(
            """
            **3. Departure Time Evaluation**  
            The engine evaluates candidate times across a ±60 min window, compares durations, and audits data coverage before advising.
            """
        )

    # Quick preview chart
    st.markdown("<div class='section-title'>Historical Journey Duration Overview</div>", unsafe_allow_html=True)
    fig = px.histogram(
        df,
        x="TotalTime",
        color="GoingTo",
        nbins=22,
        barmode="overlay",
        color_discrete_map={"GSK": "#2563EB", "Home": "#0D9488"},
        labels={"TotalTime": "Total Duration (minutes)", "GoingTo": "Destination"},
        title="Distribution of Historical Commute Durations (GSK vs Home)",
    )
    fig.update_layout(bargap=0.1, margin=dict(l=20, r=20, t=40, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 2: PREDICT TRAVEL TIME
# ==============================================================================
elif nav_choice == "Predict Travel Time":
    st.markdown("<div class='hero-title'>Predict Travel Time</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Estimate expected journey duration for a specific scheduled trip.</div>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            destination = st.selectbox(
                "Destination",
                options=["GSK", "Home"],
                index=0,
                help="GSK represents workplace in Mississauga; Home represents return commute.",
            )
            distance = st.number_input(
                "Trip Distance (km)",
                min_value=1.0,
                max_value=200.0,
                value=51.5,
                step=0.5,
                help="Historical commute corridor averages ~51 km (range: 48 to 60 km).",
            )
            day_of_week = st.selectbox(
                "Day of Week",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=1,
            )

        with col2:
            dep_time_input = st.time_input(
                "Planned Departure Time",
                value=time(7, 45),
                help="Enter 24-hr departure time. Morning (6–9 AM) typical for GSK; Afternoon/Evening for Home.",
            )
            take_407 = st.radio(
                "Take Highway 407 Toll Route Entirely?",
                options=["No", "Yes"],
                index=0,
                horizontal=True,
                help="Historically, taking Highway 407 saves ~8 minutes on average.",
            )
            model_options = list(get_available_models().keys())
            selected_model = st.selectbox(
                "Machine Learning Model",
                options=model_options,
                index=model_options.index(get_default_model_name()),
                help="AdaBoost achieved the lowest validation MAE (3.90 min).",
            )

        submit_btn = st.form_submit_button("Calculate Estimated Duration", use_container_width=True)

    if submit_btn:
        dep_str = dep_time_input.strftime("%H:%M")
        try:
            res = predict_travel_time(
                destination=destination,
                distance=distance,
                departure_time=dep_str,
                day_of_week=day_of_week,
                take_407=take_407,
                model_name=selected_model,
            )

            pred_mins = res["predicted_duration_minutes"]
            coverage = res["coverage"]

            st.markdown("---")
            st.markdown("<div class='section-title'>Prediction Summary</div>", unsafe_allow_html=True)

            res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

            with res_col1:
                st.markdown(
                    f"""
                    <div class='kpi-card' style='border-left: 4px solid #2563EB;'>
                        <div class='kpi-title'>Estimated Travel Time</div>
                        <div class='kpi-value'>{pred_mins} minutes</div>
                        <div class='kpi-sub'>Predicted using <strong>{res['model_used']}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown(
                    f"""
                    <div class='kpi-card'>
                        <div class='kpi-title'>Departure Setup</div>
                        <div class='kpi-value' style='font-size:1.3rem;'>{dep_str} ({day_of_week})</div>
                        <div class='kpi-sub'>Destination: {destination} | Toll: {take_407}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col3:
                cov_level = coverage["coverage_level"]
                badge_color = "#059669" if cov_level == "High" else ("#D97706" if cov_level == "Medium" else "#DC2626")
                st.markdown(
                    f"""
                    <div class='kpi-card'>
                        <div class='kpi-title'>Historical Data Support</div>
                        <div class='kpi-value' style='font-size:1.3rem; color:{badge_color};'>{cov_level}</div>
                        <div class='kpi-sub'>{'Well-represented' if cov_level == 'High' else 'Limited coverage'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if not coverage["is_well_supported"]:
                st.warning("⚠️ **Data Coverage Warning:** " + " ".join(coverage["warnings"]))

            st.info(
                f"ℹ️ **Interpretation:** For a {distance:.1f} km trip to {destination} departing at {dep_str} on {day_of_week} "
                f"({'with' if take_407 == 'Yes' else 'without'} Hwy 407 toll), the model estimates an average duration of "
                f"**{pred_mins} minutes**. Typical variation is approximately ±4 minutes."
            )

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")


# ==============================================================================
# PAGE 3: DEPARTURE-TIME COMPARISON
# ==============================================================================
elif nav_choice == "Departure-Time Comparison":
    st.markdown("<div class='hero-title'>Departure-Time Comparison Engine</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Compare candidate departure times across a time window to identify potentially faster departure slots.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        comp_dest = st.selectbox("Destination", ["GSK", "Home"], index=0)
        comp_dist = st.number_input("Distance (km)", min_value=10.0, max_value=150.0, value=51.5, step=0.5)
    with c2:
        comp_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], index=1)
        comp_time = st.time_input("Planned Departure Time", value=time(7, 45))
    with c3:
        comp_toll = st.radio("Toll Choice", ["No", "Yes"], index=0, horizontal=True)
        window = st.slider("Comparison Window (± minutes)", min_value=30, max_value=90, value=45, step=15)

    if st.button("Evaluate Candidate Departure Times", use_container_width=True):
        time_str = comp_time.strftime("%H:%M")
        engine = DepartureRecommendationEngine()
        rec = engine.evaluate_candidates(
            destination=comp_dest,
            distance=comp_dist,
            planned_time=time_str,
            day_of_week=comp_day,
            take_407=comp_toll,
            window_minutes=window,
            interval_minutes=15,
            compare_toll_option=True,
        )

        df_cand = rec["all_candidates"]
        rec_time = rec["recommended_time"]
        rec_dur = rec["recommended_duration_min"]
        savings = rec["estimated_savings_min"]

        st.markdown("---")

        # Recommendation Banner
        if rec["is_meaningful"]:
            st.markdown(
                f"""
                <div class='recommendation-badge'>
                    <h4 style='margin:0; font-size:1.15rem; color:#065F46;'>🌟 Suggested Departure: {rec_time}</h4>
                    <p style='margin:0.3rem 0 0 0; font-size:0.95rem;'>
                        Departing at <strong>{rec_time}</strong> is estimated at <strong>{rec_dur} min</strong> 
                        (saving ~<strong>{savings} minutes</strong> compared to {time_str} at {rec['baseline_duration_min']} min).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"ℹ️ {' '.join(rec['recommendation_notes'])}")

        if rec["warnings"]:
            st.warning("⚠️ **Coverage Notice:** " + " ".join(rec["warnings"]))

        # Visualization Chart
        st.markdown("<div class='section-title'>Candidate Departure Times vs Predicted Duration</div>", unsafe_allow_html=True)

        fig_bar = go.Figure()
        colors = [
            "#10B981" if t == rec_time else ("#F59E0B" if t == time_str else "#3B82F6")
            for t in df_cand["departure_time"]
        ]

        fig_bar.add_trace(
            go.Bar(
                x=df_cand["departure_time"],
                y=df_cand["predicted_duration_min"],
                marker_color=colors,
                text=[f"{v:.1f} m" for v in df_cand["predicted_duration_min"]],
                textposition="auto",
                name="Duration (min)",
            )
        )

        if "alt_duration_min" in df_cand.columns:
            fig_bar.add_trace(
                go.Scatter(
                    x=df_cand["departure_time"],
                    y=df_cand["alt_duration_min"],
                    mode="lines+markers",
                    line=dict(color="#8B5CF6", width=2, dash="dash"),
                    name="With Toll (407)" if comp_toll == "No" else "Without Toll",
                )
            )

        fig_bar.update_layout(
            xaxis_title="Candidate Departure Time",
            yaxis_title="Predicted Duration (minutes)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Tabular breakdown
        st.markdown("<div class='section-title'>Detailed Candidate Comparison</div>", unsafe_allow_html=True)
        display_df = df_cand.copy()
        display_df.rename(
            columns={
                "departure_time": "Departure Time",
                "predicted_duration_min": "Predicted Duration (min)",
                "delta_vs_planned_min": "Diff vs Planned (min)",
                "alt_duration_min": "Alt Toll Duration (min)",
                "toll_savings_min": "Toll Savings (min)",
            },
            inplace=True,
        )
        cols_to_show = [
            "Departure Time",
            "Predicted Duration (min)",
            "Diff vs Planned (min)",
            "Alt Toll Duration (min)",
            "Toll Savings (min)",
        ]
        st.dataframe(display_df[cols_to_show], use_container_width=True)

        st.caption(f"🛡️ {rec['disclaimer']}")


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif nav_choice == "Model Performance":
    st.markdown("<div class='hero-title'>Model Performance & Benchmarking</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Rigorous comparison of the four regression algorithms on cross-validation and test splits.</div>",
        unsafe_allow_html=True,
    )

    results_table = get_model_comparison_table()

    if results_table is not None:
        st.markdown("<div class='section-title'>Model Comparison Table</div>", unsafe_allow_html=True)
        st.dataframe(
            results_table.style.highlight_min(
                subset=["CV_MAE_Mean", "Test_MAE", "Test_RMSE"], color="#D1FAE5"
            ).highlight_max(subset=["CV_R2_Mean", "Test_R2"], color="#D1FAE5"),
            use_container_width=True,
        )

        st.markdown(
            """
            *Green highlighted values indicate top performance in each metric category. 
            Best model was selected based on 5-fold cross-validation MAE on the training set.*
            """
        )

        # Bar chart comparison
        fig_perf = px.bar(
            results_table,
            x="Model",
            y=["Test_MAE", "Test_RMSE"],
            barmode="group",
            labels={"value": "Error in Minutes", "variable": "Evaluation Metric"},
            title="Held-Out Test Error (MAE vs RMSE)",
            color_discrete_sequence=["#2563EB", "#7C3AED"],
        )
        fig_perf.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_perf, use_container_width=True)

    # Diagnostic Figures from figures folder
    figures_dir = os.path.join(os.path.dirname(__file__), "reports", "figures")
    actual_pred_img = os.path.join(figures_dir, "actual_vs_predicted.png")
    residual_img = os.path.join(figures_dir, "residual_analysis.png")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(actual_pred_img):
            st.markdown("#### Actual vs. Predicted Plots")
            st.image(actual_pred_img, caption="Actual vs Predicted across all 4 models", use_container_width=True)
    with c2:
        if os.path.exists(residual_img):
            st.markdown("#### Residual Diagnostics")
            st.image(residual_img, caption="Residual errors for the top performing model", use_container_width=True)

    # Metric Explanations
    st.markdown("<div class='section-title'>Understanding The Evaluation Metrics</div>", unsafe_allow_html=True)
    explanations = get_metric_explanations()
    for m_name, text in explanations.items():
        with st.expander(m_name, expanded=False):
            st.write(text)


# ==============================================================================
# PAGE 5: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif nav_choice == "Exploratory Data Analysis":
    st.markdown("<div class='hero-title'>Exploratory Data Analysis (EDA)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Visual investigation into commuting trends, temporal patterns, and highway toll dynamics.</div>",
        unsafe_allow_html=True,
    )

    figures_dir = os.path.join(os.path.dirname(__file__), "reports", "figures")

    eda_charts = [
        ("eda_totaltime_dist.png", "1. Journey-Time Distribution",
         "The distribution of commute duration is right-skewed with a median of 41.3 min. Several severe congestion outliers exceed 60–80 minutes."),
        ("eda_destination_comparison.png", "2. Trips & Durations by Destination",
         "Trips are evenly balanced between GSK (workplace) and Home. Both destinations have comparable average durations (~41–42 min)."),
        ("eda_hour_vs_time.png", "3. Duration by Departure Hour",
         "Trips cluster sharply in morning rush (7:00–8:00 AM) and evening rush (16:00–17:00). High delays correlate with peak morning congestion."),
        ("eda_dayofweek_vs_time.png", "4. Duration Across Days of the Week",
         "Friday exhibits noticeably shorter commute times (median ~38.4 min), while Monday and Wednesday have the highest variance and delay incidents."),
        ("eda_distance_vs_time.png", "5. Distance vs. Journey Duration & Toll Route",
         "Commutes taking Highway 407 (green) consistently finish in 30–36 minutes, saving approximately 8 minutes compared to non-toll routes (red)."),
        ("eda_correlation_heatmap.png", "6. Pre-Departure Correlation Heatmap",
         "The single strongest pre-departure predictor is Toll Usage (r = -0.45). Scheduled predictors explain ~15% of total variance."),
    ]

    for idx in range(0, len(eda_charts), 2):
        col_a, col_b = st.columns(2)
        with col_a:
            img_file, title, note = eda_charts[idx]
            img_path = os.path.join(figures_dir, img_file)
            st.markdown(f"#### {title}")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            st.caption(note)

        if idx + 1 < len(eda_charts):
            with col_b:
                img_file, title, note = eda_charts[idx + 1]
                img_path = os.path.join(figures_dir, img_file)
                st.markdown(f"#### {title}")
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                st.caption(note)


# ==============================================================================
# PAGE 6: ABOUT & METHODOLOGY
# ==============================================================================
elif nav_choice == "About & Methodology":
    st.markdown("<div class='hero-title'>About the Project & Methodology</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Detailed academic and engineering documentation of design, algorithms, and limitations.</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Overview & Objectives", "Algorithms & Preprocessing", "Limitations & Future Scope"])

    with tab1:
        st.markdown(
            """
            ### Problem Statement
            Commuters regularly face uncertainty regarding journey duration along metropolitan corridors. 
            Departure time decisions are frequently made without systematic analysis of historical commute trends. 
            This mini-project investigates how supervised machine learning can learn empirical duration patterns from 
            historical GPS-logged trips and help commuters compare candidate departure windows.

            ### Project Objectives
            1. **Clean and preprocess** raw commuter GPS logging data without data leakage.
            2. **Train and compare** exactly four foundational regression models (Linear Regression, KNN, Random Forest, AdaBoost).
            3. **Evaluate performance** honestly using MAE, RMSE, and R² across cross-validation and held-out test splits.
            4. **Construct an advisory recommendation engine** that evaluates candidate departure slots while safeguarding against unsupported queries.
            5. **Provide a professional web interface** for interactive exploration and predictions.
            """
        )

    with tab2:
        st.markdown(
            """
            ### Four Machine Learning Algorithms

            1. **Linear Regression (Baseline):**  
               Fits an ordinary least squares linear relationship between preprocessed numeric/one-hot features and total travel time.
               $$y = \\beta_0 + \\sum \\beta_i X_i$$

            2. **K-Nearest Neighbors Regressor (KNN):**  
               Finds the $k=7$ most similar historical commute days in normalized feature space and predicts duration using distance-weighted averaging.

            3. **Random Forest Regressor:**  
               Ensemble of 100 decision trees built on bootstrap samples, reducing variance and modeling non-linear interactions between departure hour, toll road usage, and destination.

            4. **AdaBoost Regressor:**  
               Sequential boosting algorithm that sequentially fits weak regressors, focusing iterative weight on previous prediction residuals.

            ### Preprocessing & Leakage Safeguards
            - **Strict Pre-Departure Features Only:** All concurrent and post-journey metrics (`AvgSpeed`, `MaxSpeed`, `MovingTime`, `FuelEconomy`, `Comments`) are strictly dropped.
            - **Reproducible Pipeline:** Imputation, standard scaling, and one-hot encoding are fit strictly on training splits within scikit-learn `Pipeline` objects.
            """
        )

    with tab3:
        st.markdown(
            """
            ### Honest Limitations
            - **No Real-Time Traffic:** The dataset consists of historical trips from 2011–2012. It does not interface with live traffic APIs (Google Maps, HERE, TomTom).
            - **Specific Route Corridor:** Data represents commutes between New Hamburg and Mississauga/Toronto via Ontario Highway 401, 403, and Highway 407 ETR. It cannot generalize to arbitrary worldwide routes.
            - **Small Sample Size:** 205 total records limit the statistical power of fine-grained hyperparameter tuning and chronological validation.
            - **Covariate Shift in Chronological Split:** In late 2011/2012, no toll trips were logged, causing negative R² on pure temporal test partitions.

            ### Future Scope
            - Integration of open transit/traffic sensor APIs (e.g. Ontario 511 open data).
            - Incorporation of real-time weather feeds (precipitation, snowfall).
            - Conformal prediction intervals to output 90% confidence duration bounds.
            """
        )

st.markdown("---")
st.caption("College ML Mini-Project | Intelligent Transportation Systems (ITS) Predictive Analytics")
