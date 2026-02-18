import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Kept as per original, though not explicitly used in Plotly dashboards
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

def simulate_monitoring_data(n_months=12, seed=42):
    """
    Simulate monthly monitoring metrics for 6 course models.
    Includes: gradual drift for one model, stable metrics for others,
    and specific performance and fairness deterioration events.
    """
    np.random.seed(seed)
    random.seed(seed) # For randint

    months = pd.date_range('2024-03-01', periods=n_months, freq='MS')

    models_config = {
        'Credit XGBoost': {'tier': 1, 'base_auc': 0.82, 'base_dir': 0.84, 'drift_rate': 0.003, 'override_rate': 0.05, 'max_validation_days': 365},
        'Trading RL Agent': {'tier': 1, 'base_auc': 0.58, 'base_dir': None, 'drift_rate': 0.005, 'override_rate': 0.15, 'max_validation_days': 365},
        'FinBERT Sentiment': {'tier': 3, 'base_auc': 0.88, 'base_dir': None, 'drift_rate': 0.001, 'override_rate': 0.02, 'max_validation_days': 1095},
        'Research Copilot': {'tier': 2, 'base_auc': None, 'base_dir': None, 'drift_rate': 0.0, 'override_rate': 0.08, 'max_validation_days': 730},
        'Rebalancing Agent': {'tier': 2, 'base_auc': None, 'base_dir': None, 'drift_rate': 0.0, 'override_rate': 0.12, 'max_validation_days': 730},
        'ESG Agent': {'tier': 2, 'base_auc': None, 'base_dir': None, 'drift_rate': 0.0, 'override_rate': 0.10, 'max_validation_days': 730}
    }

    records = []
    for model_name, config in models_config.items():
        for i, month in enumerate(months):
            # Performance (AUC)
            auc_val = None
            if config['base_auc'] is not None:
                drift = config['drift_rate'] * i
                shock = 0
                # Injected AUC drop for 'Trading RL Agent' in month 8 (index 7)
                if model_name == 'Trading RL Agent' and i >= 7:
                    shock = 0.08 # Significant drop

                # Add seasonal noise
                seasonal_noise = 0.005 * np.sin(i * np.pi / 6) # Cycle every 6 months

                auc_val = config['base_auc'] - drift - shock + np.random.normal(0, 0.005) + seasonal_noise
                auc_val = round(max(0, min(1, auc_val)), 4) # Clip between 0 and 1

            # Fairness (DIR)
            dir_val = None
            if config['base_dir'] is not None:
                fair_shock = 0
                # Injected DIR drop for 'Credit XGBoost' in month 9 (index 8)
                if model_name == 'Credit XGBoost' and i >= 8:
                    fair_shock = 0.10 # Significant drop

                # Add seasonal noise
                seasonal_noise_dir = 0.01 * np.sin(i * np.pi / 6)

                dir_val = config['base_dir'] - fair_shock + np.random.normal(0, 0.015) + seasonal_noise_dir
                dir_val = round(max(0.5, dir_val), 3) # DIR typically between 0 and 1, but often > 0.5

            # Drift (PSI) - gradual increase for models with drift_rate > 0
            psi_val = 0.05 + config['drift_rate'] * i * 3 + np.random.exponential(0.02)
            psi_val = round(min(1.0, psi_val), 3) # Cap PSI at 1.0 for realism

            # Human Override Rate
            override = config['override_rate'] + np.random.normal(0, 0.02)
            override_val = round(max(0, min(0.5, override)), 3) # Cap override rate at 50%

            # Decisions and Alerts - vary by tier
            if config['tier'] == 1:
                n_decisions = int(np.random.normal(5000, 500))
                n_alerts = np.random.poisson(3) + i # Higher alerts for critical models
            elif config['tier'] == 2:
                n_decisions = int(np.random.normal(2000, 200))
                n_alerts = np.random.poisson(2) # Moderate alerts
            else: # tier 3
                n_decisions = int(np.random.normal(500, 50))
                n_alerts = np.random.poisson(1) # Lower alerts

            # Unresolved Alerts
            unresolved_alerts_base = np.random.binomial(n_alerts, 0.3) if n_alerts > 0 else 0
            # Increase unresolved alerts for models with issues or high tier
            if model_name in ['Credit XGBoost', 'Trading RL Agent'] and i >= 8:
                 unresolved_alerts_base += random.randint(2, 5) # Increase significantly for problematic models

            unresolved_alerts = max(0, unresolved_alerts_base)


            # Last Validation - simulate a fixed validation cycle per tier
            # Tier 1: annual (365 days), Tier 2: biennial (730 days), Tier 3: triennial (1095 days)
            # Make sure last_validation is always a valid date before the current month
            validation_offset_days = np.random.randint(30, config['max_validation_days']) # Offset from current month

            # Ensure last_validation is always earlier than the current month
            last_val_date = month - timedelta(days=validation_offset_days)
            if last_val_date >= month: # If by chance it's not, set it to previous month
                last_val_date = month - timedelta(days=30)

            records.append({
                'month': month,
                'model': model_name,
                'tier': config['tier'],
                'auc': auc_val,
                'dir': dir_val,
                'psi': psi_val,
                'override_rate': override_val,
                'n_decisions': n_decisions,
                'n_alerts': n_alerts,
                'unresolved_alerts': unresolved_alerts,
                'last_validation': last_val_date
            })

    return pd.DataFrame(records)

def compute_traffic_light(row):
    """
    Assign traffic-light status based on governance thresholds for a given model's monthly metrics.
    RED = immediate action required. YELLOW = investigate. GREEN = within normal parameters.
    """
    flags = []

    # Validation cycle definitions based on tier
    # Tier 1: 12 months (365 days), Tier 2: 24 months (730 days), Tier 3: 36 months (1095 days)
    max_validation_days_per_tier = {1: 365, 2: 730, 3: 1095}
    max_days_for_tier = max_validation_days_per_tier.get(row['tier'], 365) # Default to Tier 1 if tier missing

    # 1. Performance (AUC)
    if row['auc'] is not None:
        if row['auc'] < 0.70:
            flags.append(('RED', f"AUC below 0.70 ({row['auc']:.2f})"))
        elif row['auc'] < 0.75:
            flags.append(('YELLOW', f"AUC below 0.75 ({row['auc']:.2f})"))

    # 2. Fairness (DIR)
    if row['dir'] is not None:
        if row['dir'] < 0.75:
            flags.append(('RED', f"DIR below 0.75 (4/5 violation) ({row['dir']:.2f})"))
        elif row['dir'] < 0.80:
            flags.append(('YELLOW', f"DIR below 0.80 ({row['dir']:.2f})"))

    # 3. Population Stability Index (PSI)
    if row['psi'] is not None:
        if row['psi'] > 0.25:
            flags.append(('RED', f"PSI={row['psi']:.2f} (major shift)"))
        elif row['psi'] > 0.15:
            flags.append(('YELLOW', f"PSI={row['psi']:.2f} (moderate shift)"))

    # 4. Override Rate
    if row['override_rate'] is not None:
        if row['override_rate'] > 0.25:
            flags.append(('RED', f"High override rate (>25%) ({row['override_rate']:.2%})"))
        elif row['override_rate'] > 0.15:
            flags.append(('YELLOW', f"Moderate override rate (15-25%) ({row['override_rate']:.2%})"))

    # 5. Unresolved Alerts
    if row['unresolved_alerts'] is not None:
        if row['unresolved_alerts'] > 3:
            flags.append(('RED', f"{row['unresolved_alerts']} unresolved alerts"))
        elif row['unresolved_alerts'] > 1:
            flags.append(('YELLOW', f"{row['unresolved_alerts']} unresolved alerts"))

    # 6. Validation Currency
    if row.get('last_validation') is not None and row.get('month') is not None:
        days_since_validation = (row['month'] - row['last_validation']).days

        if days_since_validation > max_days_for_tier:
            flags.append(('RED', 'Validation overdue'))
        elif days_since_validation > max_days_for_tier * 0.8:
            flags.append(('YELLOW', 'Validation due soon'))

    # Determine overall status (conservative aggregation: worst flag wins)
    if any(f[0] == 'RED' for f in flags):
        return 'RED', flags
    elif any(f[0] == 'YELLOW' for f in flags):
        return 'YELLOW', flags
    else:
        return 'GREEN', [] # No flags means green

def build_performance_fairness_dashboard(df):
    """
    Generates Plotly visualizations for AUC and DIR over time for all models.
    """
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(
                            'Model Performance (AUC) Over Time',
                            'Model Fairness (Disparate Impact Ratio) Over Time'
                        ),
                        vertical_spacing=0.1)

    # Define colors for models to keep consistency
    model_colors = {
        'Credit XGBoost': 'blue',
        'Trading RL Agent': 'red',
        'FinBERT Sentiment': 'green',
        'Research Copilot': 'purple',
        'Rebalancing Agent': 'orange',
        'ESG Agent': 'brown'
    }

    # Panel 1: AUC time series
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name].copy()
        if model_df['auc'].notna().any():
            fig.add_trace(go.Scatter(
                x=model_df['month'],
                y=model_df['auc'],
                mode='lines+markers',
                name=f"{model_name} AUC",
                line=dict(color=model_colors.get(model_name, 'gray')),
                showlegend=True
            ), row=1, col=1)

    # AUC Thresholds
    fig.add_hline(y=0.75, line_dash="dash", line_color="orange",
                  annotation_text="Yellow AUC Threshold (0.75)",
                  annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=0.70, line_dash="dash", line_color="red",
                  annotation_text="Red AUC Threshold (0.70)",
                  annotation_position="top right", row=1, col=1)

    fig.update_yaxes(title_text="AUC", range=[0.5, 1.0], row=1, col=1) # Adjust range for better visibility


    # Panel 2: Fairness (DIR) time series
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name].copy()
        if model_df['dir'].notna().any():
            fig.add_trace(go.Scatter(
                x=model_df['month'],
                y=model_df['dir'],
                mode='lines+markers',
                name=f"{model_name} DIR",
                line=dict(color=model_colors.get(model_name, 'gray')),
                showlegend=False # Only show legend for AUC to avoid clutter
            ), row=2, col=1)

    # DIR Threshold
    fig.add_hline(y=0.80, line_dash="dash", line_color="orange",
                  annotation_text="Yellow DIR Threshold (0.80)",
                  annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                  annotation_text="Red DIR Threshold (0.75)",
                  annotation_position="top right", row=2, col=1) # Adding Red DIR threshold based on problem description

    fig.update_yaxes(title_text="Disparate Impact Ratio", range=[0.5, 1.0], row=2, col=1)

    fig.update_layout(height=800, title_text='Apex Financial Group: Model Performance & Fairness Governance',
                      hovermode='x unified')
    fig.show()
    # In an app.py context, you might return fig instead of fig.show()

def build_drift_override_alerts_dashboard(df):
    """
    Generates Plotly visualizations for PSI, Human Override Rate, and Alert Trends.
    """
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=(
                            'Population Stability Index (PSI) Over Time (Selected Models)',
                            'Human Override Rate Over Time',
                            'Alert Trends (Total vs. Unresolved)'
                        ),
                        vertical_spacing=0.08)

    model_colors = {
        'Credit XGBoost': 'blue',
        'Trading RL Agent': 'red',
        'FinBERT Sentiment': 'green',
        'Research Copilot': 'purple',
        'Rebalancing Agent': 'orange',
        'ESG Agent': 'brown'
    }

    # Re-define models_config for this function's scope, as it was in the original notebook cell
    models_config = {
        'Credit XGBoost': {'tier': 1, 'base_auc': 0.82, 'base_dir': 0.84, 'drift_rate': 0.003, 'override_rate': 0.05, 'max_validation_days': 365},
        'Trading RL Agent': {'tier': 1, 'base_auc': 0.58, 'base_dir': None, 'drift_rate': 0.005, 'override_rate': 0.15, 'max_validation_days': 365},
        'FinBERT Sentiment': {'tier': 3, 'base_auc': 0.88, 'base_dir': None, 'drift_rate': 0.001, 'override_rate': 0.02, 'max_validation_days': 1095},
        'Research Copilot': {'tier': 2, 'base_auc': None, 'base_dir': None, 'drift_rate': 0.0, 'override_rate': 0.08, 'max_validation_days': 730},
        'Rebalancing Agent': {'tier': 2, 'base_auc': None, 'base_dir': None, 'drift_rate': 0.0, 'override_rate': 0.12, 'max_validation_days': 730},
        'ESG Agent': {'tier': 2, 'base_auc': None, 'base_dir': None, 'drift_rate': 0.0, 'override_rate': 0.10, 'max_validation_days': 730},
    }

    psi_models = [model for model in df['model'].unique() if models_config[model]['drift_rate'] > 0 or models_config[model]['tier'] == 1]
    for model_name in psi_models:
        model_df = df[df['model'] == model_name].copy()
        if model_df['psi'].notna().any():
            fig.add_trace(go.Bar(
                x=model_df['month'],
                y=model_df['psi'],
                name=f"{model_name} PSI",
                marker_color=model_colors.get(model_name, 'gray'),
                showlegend=True,
                opacity=0.7
            ), row=1, col=1)

    # PSI Thresholds
    fig.add_hline(y=0.15, line_dash="dash", line_color="orange",
                  annotation_text="Yellow PSI Threshold (0.15)",
                  annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="Red PSI Threshold (0.25)",
                  annotation_position="top right", row=1, col=1)

    fig.update_yaxes(title_text="PSI", range=[0, 0.5], row=1, col=1) # Adjust range for better visibility


    # Panel 4: Human Override Rate
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name].copy()
        if model_df['override_rate'].notna().any():
            fig.add_trace(go.Scatter(
                x=model_df['month'],
                y=model_df['override_rate'],
                mode='lines+markers',
                name=f"{model_name} Override",
                line=dict(color=model_colors.get(model_name, 'gray')),
                showlegend=False # Only show legend for PSI
            ), row=2, col=1)

    # Override Rate Thresholds
    fig.add_hline(y=0.15, line_dash="dash", line_color="orange",
                  annotation_text="Yellow Override Threshold (15%)",
                  annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=0.25, line_dash="dash", line_color="red",
                  annotation_text="Red Override Threshold (25%)",
                  annotation_position="top right", row=2, col=1)

    fig.update_yaxes(title_text="Override Rate", tickformat=".0%", range=[0, 0.5], row=2, col=1)


    # Panel 5: Alert Trends
    alert_monthly = df.groupby('month')['n_alerts'].sum().reset_index()
    unresolved_monthly = df.groupby('month')['unresolved_alerts'].sum().reset_index()

    fig.add_trace(go.Bar(
        x=alert_monthly['month'],
        y=alert_monthly['n_alerts'],
        name='Total Alerts',
        marker_color='lightblue',
        showlegend=True
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=unresolved_monthly['month'],
        y=unresolved_monthly['unresolved_alerts'],
        name='Unresolved Alerts',
        marker_color='darkred',
        showlegend=True
    ), row=3, col=1)

    # Unresolved Alerts Thresholds
    fig.add_hline(y=3, line_dash="dash", line_color="orange",
                  annotation_text="Yellow Unresolved Alerts Threshold (2-3)",
                  annotation_position="bottom right", row=3, col=1)
    fig.add_hline(y=5, line_dash="dash", line_color="red",
                  annotation_text="Red Unresolved Alerts Threshold (>3)",
                  annotation_position="top right", row=3, col=1) # Note: Changed to 5 for clarity, as >3 can be 4, which is ambiguous with 2-3 yellow. If 4 alerts is RED, then the line must be above 3.

    fig.update_yaxes(title_text="Number of Alerts", row=3, col=1)

    fig.update_layout(height=1000, title_text='Apex Financial Group: Model Drift, Override, and Alert Governance',
                      barmode='overlay', hovermode='x unified')
    fig.show()
    # In an app.py context, you might return fig instead of fig.show()

def build_inventory_status_dashboard(df):
    """
    Generates a Plotly visualization representing the latest traffic-light status for all models.
    """
    latest_month_df = df[df['month'] == df['month'].max()].copy()
    latest_month_df['model_tier'] = latest_month_df.apply(lambda row: f"{row['model']} (Tier {row['tier']})", axis=1)

    # Define status colors
    status_colors = {'GREEN': 'green', 'YELLOW': 'gold', 'RED': 'red'}
    latest_month_df['color'] = latest_month_df['status'].map(status_colors)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=latest_month_df['model_tier'],
        y=[1] * len(latest_month_df), # Uniform height bars for visualization
        marker_color=latest_month_df['color'],
        text=latest_month_df['status'],
        textposition='inside',
        name='Model Status'
    ))

    fig.update_layout(
        title_text='Apex Financial Group: Latest AI Model Inventory Status',
        xaxis_title="Model (Tier)",
        yaxis_title="Status Indicator",
        yaxis_visible=False, # Hide y-axis as height is arbitrary
        showlegend=False,
        height=400,
        hovermode='x unified'
    )
    fig.show()
    # In an app.py context, you might return fig instead of fig.show()

def generate_monthly_report(df, report_month=None):
    """
    Generates a text-based monthly governance report with executive summary and action items.
    """
    if report_month is None:
        report_month = df['month'].max()

    latest_month_df = df[df['month'] == report_month].copy()

    print("=" * 60)
    print(f"MONTHLY AI GOVERNANCE REPORT")
    print(f"Period: {report_month.strftime('%B %Y')}")
    print("=" * 60)

    # Executive summary
    statuses_overview = {}
    for idx, row in latest_month_df.iterrows():
        statuses_overview[row['model']] = (row['status'], row['flags'])

    n_red = sum(1 for s, _ in statuses_overview.values() if s == 'RED')
    n_yellow = sum(1 for s, _ in statuses_overview.values() if s == 'YELLOW')
    n_green = sum(1 for s, _ in statuses_overview.values() if s == 'GREEN')

    total_decisions = latest_month_df['n_decisions'].sum()
    total_alerts = latest_month_df['n_alerts'].sum()
    unresolved_alerts_count = latest_month_df['unresolved_alerts'].sum()

    print(f"\nEXECUTIVE SUMMARY:")
    print(f" Models monitored: {len(statuses_overview)}")
    print(f" GREEN: {n_green} | YELLOW: {n_yellow} | RED: {n_red}")
    print(f" Total AI decisions this month: {total_decisions:,}")
    print(f" Total alerts: {total_alerts}")
    print(f" Unresolved alerts (latest month): {unresolved_alerts_count}")


    # Action items for non-green models
    if n_red + n_yellow > 0:
        print(f"\nACTION ITEMS:")
        for model, (status, flags) in statuses_overview.items():
            if status != 'GREEN':
                print(f"\n ({status}) {model} (Tier {latest_month_df[latest_month_df['model'] == model]['tier'].iloc[0]}):")
                for severity, reason in flags:
                    print(f"   - {severity}: {reason}")

                # Define specific actions based on status and tier
                if status == 'RED':
                    print(f"   ACTION: Freeze model. Investigate within 48h. Report to CRO.")
                elif status == 'YELLOW':
                    print(f"   ACTION: Assign to model owner. Investigate findings by next meeting.")
    else:
        print(f"\nAll models are GREEN. Continued monitoring as usual.")

    print(f"\nSIGN-OFF:")
    print(f" AI Governance Officer: _________________________ Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f" CRO: _____________________________________ Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)

def run_governance_pipeline(n_months=12, seed=42):
    """
    Orchestrates the entire AI governance pipeline:
    1. Simulates monitoring data.
    2. Computes traffic light status.
    3. Generates all dashboards.
    4. Generates a monthly governance report.
    """
    print("--- Starting AI Governance Pipeline ---")

    # 1. Generate the data
    monitoring_df = simulate_monitoring_data(n_months=n_months, seed=seed)
    print(f"Monitoring data generated: {len(monitoring_df)} records, "
          f"for {monitoring_df['model'].nunique()} models, "
          f"spanning {monitoring_df['month'].nunique()} months.")
    print("\nFirst 5 rows of the monitoring data:")
    print(monitoring_df.head())

    # 2. Apply the traffic light function to each row of our monitoring data
    monitoring_df[['status', 'flags']] = monitoring_df.apply(
        lambda row: pd.Series(compute_traffic_light(row)), axis=1
    )
    print("\nTraffic-light status computed for all models across all months.")
    print("\nLatest monthly status for all models:")
    latest_month_df = monitoring_df[monitoring_df['month'] == monitoring_df['month'].max()]
    for idx, row in latest_month_df.iterrows():
        print(f"[{row['status']}] {row['model']:<25s} (Tier {row['tier']}): {', '.join([f'{s}: {r}' for s, r in row['flags']]) if row['flags'] else 'All GREEN'}")

    # 3. Build and display dashboards
    print("\n--- Generating Dashboards ---")
    build_performance_fairness_dashboard(monitoring_df.copy())
    build_drift_override_alerts_dashboard(monitoring_df.copy())
    build_inventory_status_dashboard(monitoring_df.copy())
    print("--- Dashboards Generated ---")

    # 4. Generate the monthly report
    print("\n--- Generating Monthly Governance Report ---")
    generate_monthly_report(monitoring_df.copy())
    print("--- Monthly Governance Report Generated ---")

    print("\n--- AI Governance Pipeline Completed ---")
    return monitoring_df # Optionally return the processed DataFrame

if __name__ == "__main__":
    # Example usage when run as a script:
    # This block will only execute when the file is run directly, not when imported.
    processed_data = run_governance_pipeline(n_months=12)
