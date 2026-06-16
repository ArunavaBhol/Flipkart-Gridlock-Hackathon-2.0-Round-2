import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Astram Event Data...")
df = pd.read_csv('Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')

print("2. Engineering Target Variable (Impact Duration)...")
# Calculate how long the event disrupts traffic
df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['end_time'] = pd.to_datetime(df['closed_datetime'], errors='coerce').fillna(pd.to_datetime(df['resolved_datetime'], errors='coerce'))
df['duration_minutes'] = (df['end_time'] - df['start_datetime']).dt.total_seconds() / 60.0

# Filter out anomalies and extreme outliers (events lasting more than 24 hours)
df = df[(df['duration_minutes'] > 0) & (df['duration_minutes'] <= 1440)].copy()

print("3. Feature Engineering...")
df['hour'] = df['start_datetime'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# Target Encoding for categorical variables
cat_features = ['event_cause', 'priority', 'corridor']
df[cat_features] = df[cat_features].fillna('Unknown').astype(str)

encoding_maps = {}
for col in cat_features:
    mean_enc = df.groupby(col)['duration_minutes'].mean()
    df[col + '_enc'] = df[col].map(mean_enc)
    encoding_maps[col] = mean_enc.to_dict() # Save mapping for the web app

# Save the encoding maps for live predictions later
joblib.dump(encoding_maps, 'encoding_maps.pkl')
global_mean = df['duration_minutes'].mean()
joblib.dump(global_mean, 'global_mean.pkl')

X = df[['latitude', 'longitude', 'hour_sin', 'hour_cos'] + [c + '_enc' for c in cat_features]].fillna(global_mean)
y = np.log1p(df['duration_minutes']) # Compress the target mathematically

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training the Meta-Stack (XGBoost + LightGBM)...")
base_models = [
    ('xgb', XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)),
    ('lgb', LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05, random_state=42, n_jobs=-1))
]
meta_model = Ridge(alpha=1.0)
stack = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

stack.fit(X_train, y_train)

print("5. Exporting Model for Prototype Deployment...")
joblib.dump(stack, 'traffic_impact_model.pkl')
print("Success! Model and Encoders saved.")
