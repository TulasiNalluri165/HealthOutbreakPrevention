# Modules: Cleaning → Clustering → Forecasting → Alert
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("nigeria_outbreak.csv")

df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df.dropna(subset=['report_date']) 

# Drop irrelevant columns for analysis

columns_to_drop = ['id', 'surname', 'firstname', 'middlename', 'gender_male', 'gender_female',
                   'settlement', 'age_str', 'date_of_birth', 'serotype', 'NmA', 'NmC', 'NmW']
df = df.drop(columns=columns_to_drop)

# --- 4. Aggregate Weekly Outbreak Data by State & Disease ---
disease_cols = ['cholera', 'diarrhoea', 'measles', 'viral_haemmorrhaphic_fever',
                'meningitis', 'ebola', 'marburg_virus', 'yellow_fever', 'rubella_mars', 'malaria']

df['week'] = df['report_date'].dt.to_period('W').apply(lambda r: r.start_time)

weekly_outbreak = df.groupby(['state', 'week'])[disease_cols].sum().reset_index()

# --- 5. Clustering Module: Detecting Similar Outbreak Patterns ---
# We'll cluster states based on total disease counts over time

# Sum by state
state_disease_summary = weekly_outbreak.groupby('state')[disease_cols].sum()

# Standardize
scaler = StandardScaler()
state_scaled = scaler.fit_transform(state_disease_summary)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(state_scaled)

# Add cluster labels to summary
state_disease_summary['Cluster'] = clusters

# --- 6. Visualize Clusters ---
plt.figure(figsize=(10,6))
sns.heatmap(state_disease_summary[disease_cols], cmap='YlOrRd')
plt.title("Disease Intensity per State")
plt.xlabel("Disease Type")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# --- 7. Time Series Forecasting for One Disease (e.g., Cholera in Lagos) ---
# Filter data
cholera_lagos = weekly_outbreak[weekly_outbreak['state'] == 'Lagos'][['week', 'cholera']].set_index('week')
cholera_lagos = cholera_lagos.asfreq('W', fill_value=0)

# Fit SARIMA
model = SARIMAX(cholera_lagos['cholera'], order=(1,1,1), seasonal_order=(1,1,1,52))
results = model.fit()

# Forecast next 8 weeks
forecast = results.get_forecast(steps=8)
forecast_df = forecast.summary_frame()

# --- 8. Plot Forecast ---
plt.figure(figsize=(10,5))
plt.plot(cholera_lagos.index, cholera_lagos['cholera'], label="Actual")
plt.plot(forecast_df.index, forecast_df['mean'], label="Forecast", color='red')
plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
plt.title("Cholera Forecast in Lagos (Next 8 Weeks)")
plt.xlabel("Week")
plt.ylabel("Cases")
plt.legend()
plt.grid(True)
plt.show()

# --- 9. Alert Threshold Optimization ---
# Alert if forecasted cholera > mean + 2×std
threshold = cholera_lagos['cholera'].mean() + 2 * cholera_lagos['cholera'].std()
forecast_df['Alert'] = forecast_df['mean'] > threshold

print("🚨 ALERT THRESHOLD:", round(threshold, 2))
print("\n🔔 Upcoming Alerts:")
print(forecast_df[['mean', 'Alert']])

# --- 10. Final Recommendations ---
def recommend_action(disease, state, alert):
    if alert:
        return f"🚑 High risk of {disease} outbreak in {state}. Recommend vaccination, awareness drives."
    else:
        return f"✅ No immediate threat of {disease} outbreak in {state}."

recommendations = []
for date, row in forecast_df.iterrows():
    msg = recommend_action('Cholera', 'Lagos', row['Alert'])
    recommendations.append((str(date.date()), msg))

recommendation_df = pd.DataFrame(recommendations, columns=['Week', 'Action'])
print("\n📋 Recommended Actions:")
print(recommendation_df)

# --- 11. Save Results (Optional) ---
recommendation_df.to_csv("cholera_alerts_lagos.csv", index=False)
