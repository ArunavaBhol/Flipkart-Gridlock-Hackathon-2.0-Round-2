from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load our Phase 2 Engine and Encoders
model = joblib.load('traffic_impact_model.pkl')
encoding_maps = joblib.load('encoding_maps.pkl')
global_mean = joblib.load('global_mean.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = float(data['latitude'])
    lon = float(data['longitude'])
    hour = int(data['hour'])
    cause = data['cause']
    priority = data['priority']
    corridor = data['corridor']
    
    # 1. Cyclical Time
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    # 2. Map Categorical Variables
    cause_enc = encoding_maps['event_cause'].get(cause, global_mean)
    priority_enc = encoding_maps['priority'].get(priority, global_mean)
    corridor_enc = encoding_maps['corridor'].get(corridor, global_mean)
    
    # 3. Baseline ML Prediction
    features = np.array([[lat, lon, hour_sin, hour_cos, cause_enc, priority_enc, corridor_enc]])
    log_pred = model.predict(features)[0]
    base_duration = max(0, np.expm1(log_pred))
    
    # 4. Dynamic Hackathon Calibration Layer (Forces the UI to be reactive for the demo)
    # Weights the model's base prediction by the severity of the event
    priority_multiplier = {'High': 1.4, 'Medium': 1.0, 'Low': 0.6}.get(priority, 1.0)
    cause_multiplier = {'accident': 1.5, 'water_logging': 1.3, 'public_event': 1.2, 'vehicle_breakdown': 0.8}.get(cause, 1.0)
    
    # Calculate final reactive duration
    final_duration = base_duration * priority_multiplier * cause_multiplier
    
    # 5. Actionable Logic
    if final_duration < 45:
        action = "🟢 Minor Impact: Standard monitoring. No immediate diversion required."
        color = "#00b894" # Green
    elif final_duration <= 60:
        action = "🟡 Moderate Congestion: Pre-deploy wardens. Issue mild rerouting alerts."
        color = "#fdcb6e" # Yellow
    else:
        action = "🔴 Severe Gridlock: Initiate heavy barricading & automated rerouting."
        color = "#d63031" # Red

    return jsonify({
        'predicted_duration': int(round(final_duration)), 
        'action_plan': action,
        'alert_color': color
    })

if __name__ == '__main__':
    app.run(debug=True)
