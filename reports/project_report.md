# Project Report: "Should I Leave Now?"
## ML-Based Travel Time Prediction and Departure Recommendation System

**Course/Level:** College Machine Learning Mini-Project  
**Domain:** Intelligent Transportation Systems (ITS), Machine Learning, Predictive Analytics  
**Date:** October 2026  
**Status:** Completed & Validated  

---

### Executive Summary
Urban commuters routinely face uncertainty regarding journey duration and departure timing. This project designs, implements, and evaluates an Intelligent Transportation System (ITS) predictive engine and interactive recommendation dashboard titled **"Should I Leave Now?"**. Using a real-world GPS-logged dataset of 205 commuter trips across the Ontario Highway 401, 403, and Highway 407 ETR corridor, the system predicts travel duration (in minutes) based exclusively on pre-departure information and compares candidate departure windows to assist commuter decision-making. Four core regression models—Linear Regression, K-Nearest Neighbors (KNN), Random Forest Regressor, and AdaBoost Regressor—are trained and evaluated under scikit-learn pipelines with cross-validation and held-out test splits. With a mean journey time of 41.9 minutes, the best model (AdaBoost) achieves a cross-validation Mean Absolute Error (MAE) of **3.90 minutes** (~9.3% relative error) and test MAE of **3.94 minutes**, effectively capturing systematic traffic variations and toll highway benefits without data leakage.

---

### 1. Introduction
Traffic congestion and travel-time variability pose substantial economic, environmental, and psychological burdens on daily commuters. Commuters constantly ask: *"Should I leave now, or will waiting 15–30 minutes result in a smoother, faster trip?"* Traditional navigation systems often present point estimates based on current speeds, but commuters planning departures benefit from empirical models that capture systematic temporal patterns, weekday effects, and route selection trade-offs learned from historical trips.

This mini-project develops an end-to-end machine learning solution that takes trip origin/destination, distance, day of week, scheduled departure time, and route selection, and outputs expected journey duration alongside candidate departure comparisons.

---

### 2. Problem Statement
Given a historical log of daily commuter trips containing temporal, geographic, and route indicators:
1. Predict total journey duration in minutes ($\hat{y} \in \mathbb{R}^+$) prior to departure, relying solely on information known *before* the trip commences.
2. Evaluate candidate departure times within a user-defined window (e.g., $\pm 45$ to $\pm 60$ minutes) to identify potential time savings.
3. Detect when commuter queries fall outside historical data coverage and provide explicit methodological safeguards against overconfident recommendations.

---

### 3. Objectives
- **Data Engineering & Leakage Prevention:** Clean, validate, and extract features from raw commuter GPS logs while strictly preventing target leakage from post-journey telemetry.
- **Model Training & Comparison:** Implement exactly four core regression algorithms within scikit-learn pipelines:
  1. Linear Regression (Baseline)
  2. K-Nearest Neighbors Regressor (KNN)
  3. Random Forest Regressor
  4. AdaBoost Regressor
- **Rigorous Evaluation:** Quantify performance using MAE, RMSE, and $R^2$, and contrast stratified random splitting with chronological splitting.
- **Departure Recommendation Engine:** Develop candidate departure ranking with automatic coverage auditing.
- **Interactive Web Interface:** Build a responsive Streamlit dashboard featuring prediction forms, candidate timelines, model diagnostics, and exploratory data analysis.

---

### 4. Literature Review
Research in Intelligent Transportation Systems (ITS) and machine learning provides key foundational principles for travel-time modeling:

1. **Van Lint, J. W. C., Hoogendoorn, S. P., & van Zuylen, H. J. (2005).** *Accurate freeway travel time prediction with state-space neural networks under missing data.* Transportation Research Part C: Emerging Technologies, 13(5-6), 347–369. [DOI: 10.1016/j.trc.2005.03.001](https://doi.org/10.1016/j.trc.2005.03.001)  
   *Contribution:* Demonstrates that freeway travel times exhibit non-linear state-space dynamics and highlights the importance of temporal grouping and robust missing-value handling.

2. **Mori, U., Mendiburu, A., Álvarez, M., & Lozano, J. A. (2015).** *A review of travel time estimation and forecasting for advanced traveler information systems.* Transportmetrica A: Transport Science, 11(2), 119–157. [DOI: 10.1080/23249935.2014.945674](https://doi.org/10.1080/23249935.2014.945674)  
   *Contribution:* Surveys machine learning approaches for short-term and historical travel-time estimation, distinguishing instantaneous prediction from profile-based historical models.

3. **Lippi, M., Bertini, M., & Frasconi, P. (2013).** *Short-term traffic flow forecasting: An experimental comparison of time-series analysis and supervised learning.* IEEE Transactions on Intelligent Transportation Systems, 14(2), 871–882. [DOI: 10.1109/TITS.2013.2247040](https://doi.org/10.1109/TITS.2013.2247040)  
   *Contribution:* Benchmarks classical regression and ensemble learning methods against seasonal time-series models, demonstrating the strong baseline performance of tree ensembles and regularized linear models.

4. **Zhang, Y., & Haghani, A. (2015).** *A gradient boosting method to improve travel time prediction.* Transportation Research Part C: Emerging Technologies, 58, 308–324. [DOI: 10.1016/j.trc.2015.02.019](https://doi.org/10.1016/j.trc.2015.02.019)  
   *Contribution:* Validates that gradient and adaptive boosting methods effectively capture asymmetric congestion peaks and interaction effects between departure hours and highway bottlenecks.

5. **Jenelius, E., & Koutsopoulos, H. N. (2013).** *Travel time estimation for urban road networks using low frequency probe vehicle data.* Transportation Research Part B: Methodological, 53, 64–81. [DOI: 10.1016/j.trb.2013.03.008](https://doi.org/10.1016/j.trb.2013.03.008)  
   *Contribution:* Analyzes sparse GPS probe vehicle data and articulates the challenge of covariate shift when probe coverage varies across seasons and route corridors.

*Distinction from Present Project:* While the cited academic literature often analyzes continuous sensor networks (loop detectors) or massive fleet feeds with thousands of vehicles, this mini-project focuses on a single commuter's recurring 205-trip GPS diary over 6 months, demonstrating how lightweight supervised learning can deliver practical utility while honestly addressing small-sample limitations.

---

### 5. Dataset Description
The system is built on `travel_times.csv`, a real-world dataset recorded by a commuter travelling between New Hamburg and Mississauga/Toronto via Ontario Highway 401, 403, and Highway 407 ETR toll highway between July 11, 2011 and January 6, 2012.

- **Total Observations:** 205 trips.
- **Attributes (14):** `x`, `Date`, `StartTime`, `DayOfWeek`, `GoingTo`, `Distance`, `MaxSpeed`, `AvgSpeed`, `AvgMovingSpeed`, `FuelEconomy`, `TotalTime`, `MovingTime`, `Take407All`, `Comments`.
- **Target Variable:** `TotalTime` (Total journey duration in minutes).
  - Mean: $41.90$ min, Median: $41.30$ min, Standard Deviation: $6.85$ min.
  - Min: $28.20$ min, Max: $82.30$ min.
  - Interquartile Range (IQR): $[38.40, 44.40]$ min.
- **Distance:** Mean $50.98$ km, Range $48.32$ to $60.32$ km.
- **Destinations:** GSK ($105$ morning trips) and Home ($100$ afternoon/evening trips).
- **Toll Road Usage (`Take407All`):** `No` ($170$ trips), `Yes` ($35$ trips). Mean time with 407: $35.08$ min; without 407: $43.31$ min (average savings of $\approx 8.23$ min).
- **Day of Week:** Monday ($39$), Tuesday ($48$), Wednesday ($47$), Thursday ($44$), Friday ($27$). Zero weekend trips.

---

### 6. Data Cleaning & Leakage Prevention
1. **Date & Time Parsing:** Handled dual date formats (`%m-%d-%y` and `%m/%d/%Y`) verified against the `DayOfWeek` field with 100% concordance. Converted `StartTime` into `StartHour`, `StartMinute`, and continuous `DepartureDecimal`.
2. **Physical Validity:** Confirmed all distance and duration values are positive ($> 0$).
3. **Outlier Policy:** 15 statistical outliers in `TotalTime` ($> 53.4$ min) were verified against qualitative notes in `Comments` (e.g., *"Heavy rain"*, *"Accident at 403/highway 6"*, *"Huge traffic backup"*). In accordance with ITS best practices, valid empirical delay incidents were preserved rather than arbitrarily deleted.
4. **Target Leakage Safeguard:** Post-trip telemetry (`AvgSpeed`, `AvgMovingSpeed`, `MaxSpeed`, `MovingTime`, `FuelEconomy`, `Comments`, `x`) were strictly excluded from feature matrices.

---

### 7. Feature Engineering
Pre-departure features utilized in modeling:
- **Numerical Features:** `Distance` (route distance in km), `DepartureDecimal` (continuous hour), `StartHour` (discrete hour).
- **Categorical Features:** `GoingTo` (destination: GSK, Home), `DayOfWeek` (Monday–Friday), `Take407All` (Yes, No), `TimeOfDay` (Morning, Afternoon, Evening, Night).
- **Preprocessing Pipeline:** Preprocessing is encapsulated in a scikit-learn `ColumnTransformer`:
  - `StandardScaler` applied to numerical features.
  - `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to categorical features.
  - Preprocessing parameters are fitted **strictly on training splits**.

---

### 8. Methodology & Train/Test Splitting
Two splitting strategies were systematically investigated:
1. **Stratified Split (Recommended):** An 80/20 train/test partition stratified by `Take407All` ($164$ training trips, $41$ test trips). This ensures balanced representation of toll vs non-toll trips across train and test sets.
2. **Chronological Split:** Partitioned chronologically by date (Train: July 11 – Nov 28, 2011; Test: Nov 28, 2011 – Jan 6, 2012).
   - *Covariate Shift Finding:* In the chronological test window, the driver took Highway 407 **zero times** (`Take407All == 'Yes'` is 0). In addition, Canadian winter onset increased mean travel times independently. Consequently, test $R^2$ under chronological splitting is negative ($-0.52$ to $-0.85$), though MAE remains stable ($3.51$–$3.95$ min).

---

### 9. Four-Model Comparison
All four models were trained on the identical 80/20 stratified training split ($N=164$) using 5-fold cross-validation, and evaluated on the held-out test split ($N=41$).

| Model | CV MAE (min) | CV RMSE (min) | CV $R^2$ | Test MAE (min) | Test RMSE (min) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost Regressor** | **3.90** $\pm$ 0.79 | **5.97** | **0.138** | **3.94** | **5.74** | **0.150** |
| **Random Forest Regressor** | 3.96 $\pm$ 0.89 | 6.09 | 0.093 | 3.95 | 5.73 | 0.155 |
| **Linear Regression (Baseline)** | 4.26 $\pm$ 0.73 | 5.89 | 0.156 | 3.94 | 5.77 | 0.143 |
| **KNN Regressor ($k=7$)** | 4.27 $\pm$ 0.93 | 5.83 | 0.200 | 4.48 | 6.38 | -0.049 |

*Model Selection:* **AdaBoost Regressor** was selected as the best overall model due to achieving the lowest cross-validation MAE ($3.90$ min) on training data while maintaining strong, stable generalization on held-out test data.

---

### 10. Results & Discussion
1. **Error Magnitude:** Across all models, Mean Absolute Error is bounded between **3.90 and 4.48 minutes**. On an average commute of 42 minutes, this translates to an average relative error of less than 10%, which is well within acceptable tolerance for commuter scheduling.
2. **Variance Explained ($R^2 \approx 0.15$):** Pre-departure factors (scheduled hour, weekday, destination, toll usage) explain approximately 15% of the total variance in trip times. The remaining 85% of variance stems from unobserved real-time variables: sudden traffic collisions, rain/snow intensity, lane closures, and individual traffic light timings.
3. **Key Feature Drivers:** Taking Highway 407 is the single strongest systematic time reducer ($r = -0.45$), saving an average of 8.2 minutes. Friday afternoons exhibit systematically faster travel times than midweek days.

---

### 11. Streamlit UI & Recommendation Engine
The application is deployed as a multi-page interactive dashboard (`app.py`):
- **Home / Dashboard:** Presents project KPIs, ITS context, and mandatory scientific disclaimers.
- **Predict Travel Time:** Allows commuters to configure trips and view duration estimates with confidence badges.
- **Departure-Time Comparison:** Evaluates candidate departure times across a user-selected window (e.g. $\pm 45$ min at 15-min intervals), plots duration curves, highlights the fastest departure, and displays toll savings side-by-side.
- **Model Performance:** Compares all four algorithms via tables, bar charts, actual-vs-predicted scatter plots, and residual diagnostics.
- **Exploratory Data Analysis:** Features six publication-quality analytical plots.
- **Methodology & Safeguards:** Explains underlying mathematical formulations and limitations.

---

### 12. Methodological Safeguards & Limitations
- **No Real-Time Feeds:** Predictions represent expected values based on historical patterns, not real-time GPS probe feeds.
- **Out-of-Distribution Detection:** If a user requests an evening trip to GSK or a weekend trip, the system flags the query as unsupported and presents results as exploratory.
- **Sample Size:** 205 records from a single commuter corridor limit global generalizability.

---

### 13. Future Scope
1. **Multi-Corridor Scaling:** Incorporate open probe datasets (e.g., Caltrans PeMS or Ontario 511 open APIs).
2. **Weather Integration:** Augment training features with historical precipitation, snow accumulation, and visibility data.
3. **Conformal Prediction:** Implement distribution-free prediction intervals (e.g., 90% confidence duration bounds) to convey uncertainty to commuters.

---

### 14. Conclusion
The "Should I Leave Now?" mini-project demonstrates that supervised machine learning can learn reproducible travel-time patterns from historical commuter diaries without data leakage. The AdaBoost and Random Forest models achieve an average prediction error under 4 minutes, and the recommendation engine provides intuitive departure window comparisons with transparent safeguards.

---

### 15. References
1. Van Lint, J. W. C., Hoogendoorn, S. P., & van Zuylen, H. J. (2005). *Accurate freeway travel time prediction with state-space neural networks under missing data.* Transportation Research Part C: Emerging Technologies, 13(5-6), 347–369. [DOI: 10.1016/j.trc.2005.03.001](https://doi.org/10.1016/j.trc.2005.03.001)
2. Mori, U., Mendiburu, A., Álvarez, M., & Lozano, J. A. (2015). *A review of travel time estimation and forecasting for advanced traveler information systems.* Transportmetrica A: Transport Science, 11(2), 119–157. [DOI: 10.1080/23249935.2014.945674](https://doi.org/10.1080/23249935.2014.945674)
3. Lippi, M., Bertini, M., & Frasconi, P. (2013). *Short-term traffic flow forecasting: An experimental comparison of time-series analysis and supervised learning.* IEEE Transactions on Intelligent Transportation Systems, 14(2), 871–882. [DOI: 10.1109/TITS.2013.2247040](https://doi.org/10.1109/TITS.2013.2247040)
4. Zhang, Y., & Haghani, A. (2015). *A gradient boosting method to improve travel time prediction.* Transportation Research Part C: Emerging Technologies, 58, 308–324. [DOI: 10.1016/j.trc.2015.02.019](https://doi.org/10.1016/j.trc.2015.02.019)
5. Jenelius, E., & Koutsopoulos, H. N. (2013). *Travel time estimation for urban road networks using low frequency probe vehicle data.* Transportation Research Part B: Methodological, 53, 64–81. [DOI: 10.1016/j.trb.2013.03.008](https://doi.org/10.1016/j.trb.2013.03.008)
