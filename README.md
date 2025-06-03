# KOVAI.CO-TASK

Key Insights from the Dataset
1.	Highest Average Ridership:
                              The Local Route and Light Rail services showed the highest average daily passenger journeys, indicating their central role in daily commuting.

2.	Weekly Usage Patterns:
                              Weekdays generally had higher ridership across all services. School ridership showed a sharp decline on weekends, while Peak Service usage was concentrated during working days.


3.  Peak Usage Days:
                                 Local Route & Light Rail peaked on Wednesdays and Thursdays.
                                 School Services saw their highest usage on Tuesdays.

4.  Service Correlation:
                          Strong positive correlations were observed between Local Route, Rapid Route, and Light Rail, suggesting that their usage patterns move in tandem and may serve similar commuter groups.
5.
  Variability in Ridership:
                          School and Peak Services showed high standard deviation, suggesting irregular usage patterns possibly influenced by holidays and school schedules.
























 
Forecasting Algorithm Used: XGBoost Regressor:
➤ Overview:
XGBoost (Extreme Gradient Boosting) is an efficient and scalable implementation of gradient boosting framework. It's suitable for time series regression due to its ability to model non-linear relationships and handle missing or skewed data.
➤ Model Features:
•	Day_of_Week: Encodes which day the data point falls on (0=Monday to 6=Sunday)
•	Is_Weekend: Binary feature to capture weekend effects
•	Lag_1 and Lag_7: Previous day and previous week’s ridership values used as predictive inputs
➤ Model Parameters:
Parameter	Value	Description
n_estimators	200	Number of trees in the model
learning_rate	0.1	Step size shrinkage for each boosting step
max_depth	4	Maximum depth of each tree (controls overfitting)
random_state	42	Ensures reproducible results
➤ Evaluation Metric:
•	MAPE (Mean Absolute Percentage Error) was used to evaluate prediction accuracy.
o	Local Route MAPE: ~1.44%
o	Light Rail MAPE: ~0.69%
o	Rapid Route MAPE: ~1.29

 Conclusion
           The XGBoost-based forecasting system accurately predicted future ridership across key transport services using day-wise features and lag-based trends. These insights and forecasts can aid city planners in optimizing route schedules, managing operational loads, and planning for infrastructure improvements.

