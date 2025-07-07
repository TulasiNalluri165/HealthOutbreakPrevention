import pandas as pd
import numpy as np
from datetime import datetime

# 📥 Load your dataset
df = pd.read_csv("nigeria_outbreak.csv")

# 📆 Convert report_date to datetime
df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')

# ✅ Filter out rows with invalid dates
df = df.dropna(subset=['report_date'])

# 🧪 List of disease columns (you can adjust based on your dataset)
disease_cols = [
    'cholera', 'diarrhoea', 'measles', 'meningitis', 'ebola', 
    'marburg_virus', 'yellow_fever', 'rubella_mars', 'malaria'
]

# 📂 Prepare to collect alerts
alerts = []

# 🔁 Loop through states and diseases
for state in df['state'].unique():
    state_df = df[df['state'] == state].copy()

    for disease in disease_cols:
        # 🧼 Skip if disease column not in dataset
        if disease not in state_df.columns:
            continue

        # ⚠️ Skip if there are NO cases at all (we allow even 1+ case to process)
        if len(state_df[state_df[disease] > 0]) < 1:
            print(f"⏩ Skipping {state} - {disease} due to no reported cases.")
            continue

        print(f"📈 Forecasting {disease} in {state}...")

        # 📊 Group by week for time series
        ts = state_df[['report_date', disease]].copy()
        ts = ts.groupby('report_date').sum().asfreq('W').fillna(0)

        # 🔮 Simple forecasting using rolling average
        ts['forecast'] = ts[disease].rolling(window=2, min_periods=1).mean().shift(1)

        # 📆 Latest prediction
        latest_date = ts.index.max()
        predicted_value = ts.loc[latest_date, 'forecast']

        # 🚨 Trigger alert if forecast exceeds minimal threshold
        if predicted_value > 0:
            print(f"🚨 ALERT: Possible {disease} outbreak in {state} (Predicted: {predicted_value:.2f})")
            alerts.append({
                "state": state,
                "disease": disease,
                "forecast_date": latest_date.strftime('%Y-%m-%d'),
                "predicted_cases": round(predicted_value, 2)
            })

# 📤 Save all alerts to CSV
alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv("all_disease_alerts.csv", index=False)
print("✅ Forecasting completed. Alerts saved to 'all_disease_alerts.csv'.")
