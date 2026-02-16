
# AI Model Governance Monitoring Dashboard: A CFA's Workflow for Risk Oversight

## Case Study: Ensuring Responsible AI in Financial Applications

### Introduction: The AI Governance Imperative

As a CFA Charterholder and Investment Professional at **"Apex Financial Group"**, your role extends beyond traditional portfolio management to ensuring the ethical and compliant use of Artificial Intelligence (AI) models. In today's dynamic financial landscape, AI models are increasingly deployed for critical tasks like credit scoring, algorithmic trading, and sentiment analysis. However, with their power comes significant responsibility. The **AI Governance Committee** at Apex Financial Group, of which you are a key member, meets monthly to rigorously monitor these models.

The primary goal is to ensure that AI models operate within established risk and fairness parameters, adhere to regulatory standards (like SR 11-7 for ongoing monitoring), and maintain their integrity over time. Without effective oversight, models can silently degrade, leading to financial losses, reputational damage, and regulatory penalties.

This Jupyter Notebook simulates a crucial real-world workflow: accessing and analyzing a multi-panel monitoring dashboard, identifying deviations, and generating a governance report. This process transforms raw monitoring data into actionable governance intelligence, providing an "early warning system" for your committee's "cockpit." You'll be performing tasks that directly support **CFA Standard V(A): Diligence and Reasonable Basis**, by ensuring continuous scrutiny of AI models.

---

### 1. Setting Up the Monitoring Environment

Before we can dive into analyzing model performance, we need to ensure our environment is prepared and that we have the necessary tools and simulated data for our governance monitoring dashboard.

#### 1.1. Installing Required Libraries

We'll use `pandas` and `numpy` for data manipulation, `matplotlib` for basic plotting, and `plotly` for interactive visualizations, which are essential for a comprehensive dashboard.

```python
!pip install pandas numpy matplotlib plotly
```

#### 1.2. Importing Dependencies

Here, we import all the necessary Python libraries that will be used throughout our analysis and dashboard creation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
```

---

### 2. Simulating 12 Months of AI Model Monitoring Data

**Story + Context + Real-World Relevance:**
As a CFA, you understand that governance requires continuous data. To simulate a real-world scenario for the AI Governance Committee, we need to generate a synthetic dataset representing 12 months of monitoring metrics for six distinct AI models used by Apex Financial Group. This data will include various performance indicators, fairness metrics, and operational statistics. It's crucial to simulate realistic noise, seasonal variations, and critically, inject specific deterioration events to test the effectiveness of our monitoring system. This allows the committee to practice identifying and responding to real-world challenges.

The generated data includes:
*   `month`: The monitoring period.
*   `model`: Name of the AI model (e.g., 'Credit XGBoost', 'Trading RL Agent').
*   `tier`: Model criticality tier, influencing validation frequency.
*   `auc`: Area Under the Receiver Operating Characteristic (ROC) curve, a key performance metric.
*   `dir`: Disparate Impact Ratio, a fairness metric.
*   `psi`: Population Stability Index, indicating input data drift.
*   `override_rate`: Rate at which human analysts override model recommendations.
*   `n_decisions`: Number of decisions made by the model.
*   `n_alerts`: Number of internal system alerts generated.
*   `unresolved_alerts`: Number of alerts still pending resolution.
*   `last_validation`: Date of the model's last thorough validation.

**Injected Events for Testing Governance:**
To truly test our monitoring framework, we'll inject two critical events:
1.  **Performance Deterioration:** The 'Trading RL Agent' will experience a significant drop in `auc` in month 8, simulating a regime change or a market shift it's not adapting to.
2.  **Fairness Deterioration:** The 'Credit XGBoost' model will show a drop in `dir` in month 9, simulating a shift in borrower demographics or an issue with a protected attribute.

These simulated events allow us to confirm that our dashboard can effectively detect and highlight issues, mimicking the "early warning system" role of continuous monitoring required by financial regulations.

```python
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
        'ESG Agent': {'tier': 2, 'base_auc': None, 'base_dir': None, 'drift_rate': 0.0, 'override_rate': 0.10, 'max_validation_days': 730},
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

# Generate the data
monitoring_df = simulate_monitoring_data()

print(f"Monitoring data generated: {len(monitoring_df)} records, "
      f"for {monitoring_df['model'].nunique()} models, "
      f"spanning {monitoring_df['month'].nunique()} months.")
print("\nFirst 5 rows of the monitoring data:")
print(monitoring_df.head())
```

**Explanation of Execution:**
The output confirms the successful generation of a comprehensive synthetic dataset. This table, `monitoring_df`, serves as the raw input a CFA would receive from various model monitoring systems. It's now ready for processing by our governance framework. The simulated events (AUC drop for 'Trading RL Agent' and DIR drop for 'Credit XGBoost') are now embedded within this data, awaiting detection by our governance tools.

---

### 3. Implementing the AI Governance Traffic-Light Status Engine

**Story + Context + Real-World Relevance:**
For the AI Governance Committee, quickly understanding the health of each model is paramount. Reviewing raw metrics for six models over 12 months is inefficient. As a CFA, you need a concise, actionable summary. This is where the "Traffic-Light Status Engine" comes in. It translates complex metrics into an easily digestible 'GREEN', 'YELLOW', or 'RED' status, akin to a control panel in a cockpit.

This engine implements the governance framework's thresholds across multiple dimensions: performance (AUC), fairness (Disparate Impact Ratio), data drift (PSI), human oversight (override rate), operational efficiency (unresolved alerts), and regulatory compliance (validation currency). The core principle for aggregation is **conservatism**: the worst individual flag determines the overall model status, ensuring that any potential issue immediately draws the committee's attention. This aligns with financial risk management principles where potential downsides are prioritized.

**Traffic-Light Thresholds:**

| Metric               | GREEN                                 | YELLOW                                 | RED                                   |
| :------------------- | :------------------------------------ | :------------------------------------- | :------------------------------------ |
| **AUC**              | $\ge 0.75$                            | $0.70 - 0.75$                          | $< 0.70$                              |
| **Disparate Impact Ratio** | $\ge 0.80$                            | $0.75 - 0.80$                          | $< 0.75$                              |
| **PSI (drift)**      | $< 0.15$                              | $0.15 - 0.25$                          | $> 0.25$                              |
| **Override Rate**    | $< 15\%$                              | $15\% - 25\%$                          | $> 25\%$                              |
| **Unresolved Alerts**| $< 1$                                 | $2 - 3$                                | $> 3$                                 |
| **Validation Currency**| Within cycle                          | $> 80\%$ of cycle                      | Overdue                               |

The `compute_traffic_light` function will evaluate each metric against these thresholds. If any metric crosses a RED threshold, the model's overall status is RED. If no RED flags but any YELLOW flags are present, the status is YELLOW. Otherwise, it's GREEN. This "worst flag wins" aggregation ensures maximum vigilance.

```python
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

# Apply the traffic light function to each row of our monitoring data
monitoring_df[['status', 'flags']] = monitoring_df.apply(
    lambda row: pd.Series(compute_traffic_light(row)), axis=1
)

print("Traffic-light status computed for all models across all months.")
print("\nLatest monthly status for all models:")
latest_month_df = monitoring_df[monitoring_df['month'] == monitoring_df['month'].max()]
for idx, row in latest_month_df.iterrows():
    print(f"[{row['status']}] {row['model']:<25s} (Tier {row['tier']}): {', '.join([f'{s}: {r}' for s, r in row['flags']]) if row['flags'] else 'All GREEN'}")

```

**Explanation of Execution:**
The output shows the calculated traffic-light status for each model for the latest month, along with the specific reasons (flags) for non-green statuses. This immediate summary is vital for the AI Governance Committee. For instance, if 'Trading RL Agent' shows a 'RED' status due to "AUC below 0.70", it immediately signals a critical performance issue that requires urgent attention. This system streamlines the identification of problems, allowing the CFA to prioritize investigations and allocate resources effectively, preventing silent degradation.

---

### 4. Visualizing Model Performance and Fairness Trends

**Story + Context + Real-World Relevance:**
As a CFA, understanding the historical trends of key metrics is crucial for identifying patterns, validating the effectiveness of interventions, and predicting potential future issues. The raw data and even the traffic-light status only show a snapshot. Visualizations provide the narrative of a model's health over time. This section focuses on performance (AUC) and fairness (Disparate Impact Ratio), which are fundamental to both fiduciary duty and regulatory compliance. Declining AUC could mean financial underperformance, while a deteriorating DIR raises ethical and legal concerns. You, as a CFA, need to be able to spot these trends immediately.

**Concepts:**
*   **Area Under the Curve (AUC):** Measures a model's ability to distinguish between classes (e.g., good vs. bad credit risks). A higher AUC ($AUC \in [0,1]$) indicates better performance.
*   **Disparate Impact Ratio (DIR):** Quantifies fairness by comparing the selection rates of different demographic groups. For example, if $P(Y=1|D=unprivileged)$ is the probability of a positive outcome for the unprivileged group and $P(Y=1|D=privileged)$ for the privileged group, then:
    $$ DIR = \frac{P(Y=1|D=unprivileged)}{P(Y=1|D=privileged)} $$
    A $DIR$ close to 1 indicates fairness, while values significantly below 1 (e.g., $<0.8$) suggest disparate impact against the unprivileged group.

These visualizations will include horizontal lines representing our defined YELLOW and RED thresholds, allowing for immediate visual assessment against governance standards.

```python
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

# Execute the dashboard creation for performance and fairness
build_performance_fairness_dashboard(monitoring_df)
```

**Explanation of Execution:**
The generated interactive plots vividly illustrate the performance (AUC) and fairness (DIR) trends for each model over the 12-month period. For example, a CFA would immediately notice the sharp drop in AUC for the 'Trading RL Agent' around month 8, crossing the yellow and potentially red thresholds. Similarly, the 'Credit XGBoost' DIR plot might show a decline in month 9, indicating a potential fairness issue. These visual cues are invaluable for the AI Governance Committee to quickly identify models requiring deeper investigation, allowing for proactive risk mitigation. This direct visual evidence informs decisions far more effectively than static reports alone.

---

### 5. Analyzing Data Drift, Human Overrides, and Alert Trends

**Story + Context + Real-World Relevance:**
Beyond direct performance and fairness, the health of an AI model is also dictated by the stability of its input data, the level of human trust (or distrust), and the efficiency of operational support. For a CFA, data drift can signal a fundamental change in market conditions or customer behavior, invalidating the model's original training assumptions. High human override rates could indicate the model is no longer fit-for-purpose or that its recommendations are not trusted. An increasing backlog of unresolved alerts suggests operational inefficiencies or underlying systemic issues. Monitoring these metrics provides critical insights into the operational integrity and continued relevance of the AI models.

**Concepts:**
*   **Population Stability Index (PSI):** Measures how much a population (e.g., input data distribution) has shifted over time. A low $PSI$ (e.g., $<0.1$) suggests stability, while a high $PSI$ ($>0.25$) indicates a significant shift that could degrade model performance.
    $$ PSI = \sum_{i=1}^{n} (\text{Actual}_i - \text{Expected}_i) \ln \left( \frac{\text{Actual}_i}{\text{Expected}_i} \right) $$
    where $\text{Actual}_i$ and $\text{Expected}_i$ are the percentages of records in bin $i$ for the current and baseline periods, respectively.
*   **Human Override Rate:** The proportion of model-generated decisions that are manually altered by human operators. A rising rate is an "underappreciated signal" that may indicate model degradation or miscalibration, even before performance metrics like AUC detect it.
*   **Alert Trends:** Tracking total alerts and, more critically, unresolved alerts helps assess the operational burden and responsiveness of the model support team.

These panels provide a holistic view of model governance, addressing both quantitative and qualitative aspects of AI model health.

```python
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

    # Panel 3: PSI (Drift) for selected models (those with potential drift issues or high tier)
    # Using bar charts for PSI as it's often reported per month/period bin
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

# Execute the dashboard creation for drift, overrides, and alerts
build_drift_override_alerts_dashboard(monitoring_df)
```

**Explanation of Execution:**
The visualizations clearly show trends in data drift (PSI), human override rates, and alert management. A CFA would use these to identify:
*   **PSI spikes:** For instance, if 'Credit XGBoost' shows a sustained increase in PSI, it suggests that the characteristics of new credit applicants might have changed significantly, impacting the model's predictive power.
*   **Rising override rates:** A consistently high or increasing human override rate for 'Trading RL Agent' could indicate that human traders are losing trust in the model's recommendations, potentially due to market shifts the model hasn't adapted to. This is an early warning signal, possibly preceding a drop in AUC.
*   **Unresolved alerts:** A growing red bar for 'Unresolved Alerts' might indicate that the model support team is overwhelmed or that issues are not being addressed promptly, posing operational risks.

These insights are critical for managing the operational risks associated with AI models, ensuring they remain relevant and maintain the trust of human operators.

---

### 6. Summarizing Model Inventory Health with Traffic-Light Status

**Story + Context + Real-World Relevance:**
The AI Governance Committee needs a high-level overview of the entire AI model inventory's health at a glance. After reviewing detailed trends, the most critical piece of information is the current "traffic-light" status for each model. This summary acts as the final "cockpit" display, providing immediate actionable intelligence: which models are GREEN (healthy), YELLOW (under investigation), or RED (requiring immediate action). For a CFA, this directly informs the committee's agenda, allowing them to allocate discussion time and resources to the most pressing issues. This visualization quickly answers the question: "Are all models accounted for and within their review cycle, and what is their current health?"

```python
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

# Execute the dashboard creation for model inventory status
build_inventory_status_dashboard(monitoring_df)
```

**Explanation of Execution:**
This bar chart provides an instant overview of the entire model inventory's health. A CFA can immediately see how many models are green, yellow, or red. This visual summary is crucial for the AI Governance Committee's monthly meeting. For example, if two models are flagged 'RED', the committee's agenda will immediately prioritize these models, triggering incident response protocols (e.g., "freeze model, investigate within 48h, report to CRO"). This dashboard effectively transforms complex monitoring data into clear, actionable governance intelligence, aligning perfectly with the committee's need for a concise "cockpit" view.

---

### 7. Generating the Monthly AI Governance Report

**Story + Context + Real-World Relevance:**
The AI Governance Committee's review culminates in a formal monthly report. As a CFA, your responsibility includes not only identifying issues but also documenting them, summarizing findings for executive leadership, and explicitly defining action items. A dashboard is an excellent monitoring tool, but a structured report is the formal record for compliance, audits, and strategic decision-making. This report provides an executive summary, clearly stating the overall health of the model inventory (counts of Green/Yellow/Red models), and crucially, details specific action items for all non-green models. This ensures transparency, accountability, and a clear path for remediation, fulfilling regulatory requirements for ongoing monitoring and incident management.

```python
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

# Generate the report for the latest month
generate_monthly_report(monitoring_df)
```

**Explanation of Execution:**
The generated text report provides a clear, concise summary of the AI model inventory's status for the most recent month. It highlights the total number of models in each traffic-light category and, crucially, lists specific action items for any models flagged as 'YELLOW' or 'RED'. For instance, seeing "Trading RL Agent" as RED with an "AUC below 0.70" flag, followed by the action "Freeze model. Investigate within 48h. Report to CRO," provides the clear directive a CFA needs to initiate the incident response protocol. This report is the official output of the AI Governance Committee's monthly review, ensuring accountability and driving timely corrective measures, a core aspect of responsible AI governance in finance. It directly supports regulatory compliance by providing documented evidence of ongoing oversight and action.

---
