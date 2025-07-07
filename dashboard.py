import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Load forecast alerts and full dataset
df = pd.read_csv("all_disease_alerts.csv")
df_full = pd.read_csv("nigeria_outbreak.csv")
df_full['report_date'] = pd.to_datetime(df_full['report_date'], errors='coerce')
df_full = df_full.dropna(subset=['report_date'])

disease_cols = [
    'cholera', 'diarrhoea', 'measles', 'meningitis', 'ebola',
    'marburg_virus', 'yellow_fever', 'rubella_mars', 'malaria'
]

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Nigeria Outbreak Forecast"

app.layout = html.Div([
    html.H1("Nigeria Health Outbreak Dashboard", style={'textAlign': 'center', 'color': '#003366'}),

    html.Div([
        html.Div([
            html.Label("Select Disease:"),
            dcc.Dropdown(
                id='disease-dropdown',
                options=[{'label': d.title(), 'value': d} for d in disease_cols],
                value='cholera',
                clearable=False
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Select State:"),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': s, 'value': s} for s in sorted(df['state'].unique())],
                value=df['state'].iloc[0],
                clearable=False
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    html.Div([
        html.Button("Download Alerts CSV", id="btn-download", n_clicks=0, className="btn btn-primary"),
        dcc.Download(id="download-alerts"),
    ], style={'padding': '10px'}),

    html.Div([
        html.H3("Disease Forecast Alerts"),
        dcc.Graph(id='map-plot')
    ], style={'padding': '10px'}),

    html.Div([
        html.H3(id='time-series-title'),
        dcc.Graph(id='time-series-plot')
    ], style={'padding': '10px'}),
])

@app.callback(
    Output('map-plot', 'figure'),
    Output('time-series-plot', 'figure'),
    Output('time-series-title', 'children'),
    Output("download-alerts", "data"),
    Input('disease-dropdown', 'value'),
    Input('state-dropdown', 'value'),
    Input("btn-download", "n_clicks"),
    prevent_initial_call='initial_duplicate'
)
def update_dashboard(selected_disease, selected_state, n_clicks):
    # Filter forecast for selected disease
    filtered_df = df[df['disease'] == selected_disease]

    if filtered_df.empty:
        map_fig = px.scatter_geo()
        map_fig.update_layout(title="⚠️ Forecast data unavailable.")
    else:
        map_fig = px.scatter_geo(
            filtered_df,
            locations="state",
            locationmode="country names",
            color="predicted_cases",
            hover_name="state",
            projection="natural earth",
            title=f"{selected_disease.title()} Alerts in Nigeria",
            size_max=10,
            color_continuous_scale="Reds",
        )
        map_fig.update_geos(
            scope="africa",
            center={"lat": 9.0820, "lon": 8.6753},
            projection_scale=6,
            showcoastlines=True,
            showcountries=True,
            countrycolor="Black"
        )
        map_fig.update_layout(height=500)

    # Filter original dataset
    state_df = df_full[df_full['state'] == selected_state]

    if selected_disease not in state_df.columns or state_df[selected_disease].sum() == 0:
        ts_fig = px.line(title="⚠️ Forecast data unavailable.")
    else:
        ts = state_df[['report_date', selected_disease]].copy()
        ts = ts.groupby('report_date').sum().asfreq('W').fillna(0).reset_index()
        ts_fig = px.line(
            ts,
            x='report_date',
            y=selected_disease,
            labels={'report_date': 'Date', selected_disease: 'Cases'},
            title=None
        )
        ts_fig.update_traces(line=dict(color='blue'))
        ts_fig.update_layout(height=500)

    title = f"{selected_disease.title()} Cases Over Time in {selected_state}"

    # Download handler
    if n_clicks:
        return map_fig, ts_fig, title, dcc.send_data_frame(filtered_df.to_csv, f"{selected_disease}_{selected_state}_forecast.csv", index=False)

    return map_fig, ts_fig, title, dash.no_update

# Run app
if __name__ == '__main__':
    app.run(debug=True)
