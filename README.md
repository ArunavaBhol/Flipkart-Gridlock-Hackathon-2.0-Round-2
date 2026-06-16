# Flipkart-Gridlock-Hackathon-2.0-Round-2
An AI-driven proactive command dashboard that forecasts event-induced traffic gridlock in Bengaluru and generates automated rerouting protocols before the congestion happens.

# The Vision
Urban mobility in Bengaluru often relies on reactive traffic management—clearing gridlock only after it has already paralyzed a corridor. The Astram Command Center was built to shift that paradigm to proactive forecasting. By analyzing historical event data, this prototype allows city planners to simulate high-stress scenarios (like planned rallies or sudden breakdowns) and know exactly how long the disruption will last, allowing them to pre-deploy wardens and divert traffic in advance.


# The AI Engine (Meta-Stack Architecture)
Built completely from scratch, the predictive engine leverages an advanced Machine Learning Meta-Stack:

Cyclical Time Engineering: Transformed raw hours into continuous sine and cosine waves, allowing the model to understand the continuous loop of time (e.g., 11:00 PM and 12:00 AM).
Target Encoding & Spatial Logic: Mapped categorical variables and geohash locations to their historical congestion weights to give the model localized neighborhood intelligence.
Ensemble Learning: Blended LightGBM (optimized for heavy statistical aggregations) and XGBoost across a 5-Fold Cross-Validation loop to ensure robust, highly generalized predictions without overfitting.


# The Prototype Build
The mathematical model was exported and wrapped in a lightweight Flask web framework, bridging the backend AI with a sleek, responsive HTML/CSS dashboard styled in Flipkart's signature brand colors. It features a Dynamic Calibration Layer that weights the model's base predictions by event severity, providing traffic police with actionable, color-coded rerouting protocols in real-time.
