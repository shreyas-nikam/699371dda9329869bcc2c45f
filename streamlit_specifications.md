
# Streamlit Application Specification: AI Governance Monitoring Dashboard

## 1. Application Overview

**Purpose:**
The "Apex Financial Group: AI Governance Monitoring Dashboard" is a Streamlit application designed for CFA Charterholders and Investment Professionals to provide operational oversight of AI models. It addresses the critical need for transparent, measurable, and actionable AI governance in finance, ensuring models operate within ethical, performance, and compliance parameters (e.g., SR 11-7, CFA Standard V(A)). The dashboard serves as the AI Governance Committee's "cockpit," transforming complex monitoring data into immediate, actionable insights.

**High-Level Story Flow:**

1.  **Introduction & Context (Overview Page):** The user, a CFA Charterholder, is introduced to the imperative of AI governance at Apex Financial Group. This section frames the dashboard's role within their fiduciary duties and regulatory requirements, explaining the simulated data and the underlying traffic-light status engine.
2.  **Interactive Dashboard (Dashboard Page):** The CFA navigates to the main dashboard to visualize historical trends and current health for six key AI models across multiple dimensions: performance (AUC), fairness (Disparate Impact Ratio), data drift (PSI), human override rates, and alert trends. A traffic-light summary provides an at-a-glance health status for the entire model inventory. The interactive plots allow for drill-down and identification of specific deterioration events (e.g., AUC drop for 'Trading RL Agent', DIR drop for 'Credit XGBoost').
3.  **Actionable Governance Report (Governance Report Page):** Based on the dashboard's findings, the CFA generates a formal monthly governance report. This report summarizes the overall model inventory health, details specific issues for 'YELLOW' or 'RED' flagged models, and outlines clear, predefined action items, serving as an auditable record for compliance and incident response protocols. This culminates the workflow by providing documented evidence of oversight and driving remediation.

## 2. Code Requirements

### Import Statement

The following import statement will be at the top of `app.py`:

```python
from source import *
import streamlit as st
import pandas as pd
from datetime import datetime
import io
from contextlib import redirect_stdout # For capturing print output
```

### `st.session_state` Design

`st.session_state` will be used to preserve the monitoring data and the current page selection across user interactions.

*   **`st.session_state.monitoring_df`**:
    *   **Initialization**: Checked on app startup. If not present, `simulate_monitoring_data()` is called once to generate the core dataset.
    *   **Update**: After `monitoring_df` is generated, `compute_traffic_light()` is applied to each row to add `status` and `flags` columns. This DataFrame then contains all the necessary data and pre-computed governance statuses.
    *   **Read**: This DataFrame is passed as input to all dashboard building and report generation functions from `source.py`.
    *   **Purpose**: Ensures data is loaded and processed only once, maintaining consistency and efficiency.

*   **`st.session_state.page`**:
    *   **Initialization**: Initialized to `"Overview"` on app startup.
    *   **Update**: Updated based on the user's selection from the `st.sidebar.selectbox`. An `st.experimental_rerun()` is triggered to immediately switch the displayed page.
    *   **Read**: Controls the conditional rendering logic in the main content area to display the selected page.
    *   **Purpose**: Manages multi-page navigation within the single `app.py` file.

### UI Interactions and Function Calls

The application will use a sidebar for navigation and the main area for content display.

#### Application Structure

```python
# app.py pseudo-code
import streamlit as st
import pandas as pd
from datetime import datetime
from source import * # Assumes all functions are directly importable
import io
from contextlib import redirect_stdout

st.set_page_config(layout="wide", page_title="AI-Finance Governance Dashboard")

# --- Session State Initialization ---
if 'monitoring_df' not in st.session_state:
    st.session_state.monitoring_df = simulate_monitoring_data()
    st.session_state.monitoring_df[['status', 'flags']] = st.session_state.monitoring_df.apply(
        lambda row: pd.Series(compute_traffic_light(row)), axis=1
    )

if 'page' not in st.session_state:
    st.session_state.page = "Overview"

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    ["Overview", "Dashboard", "Governance Report"],
    index=["Overview", "Dashboard", "Governance Report"].index(st.session_state.page)
)
if page_selection != st.session_state.page:
    st.session_state.page = page_selection
    st.experimental_rerun()

# --- Main Content Area ---
st.title("🛡️ Apex Financial Group: AI Governance Monitoring Dashboard")
st.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Conditional rendering based on st.session_state.page
# ... (see page-specific details below)
```

#### Page-Specific Content and Interactions

**1. "Overview" Page**

*   **Content**: Provides a foundational understanding of the AI governance imperative, the simulated data context, and the traffic-light status framework.
*   **Widgets/Markdown**:
    ```python
    st.header("The AI Governance Imperative for Investment Professionals")
    st.markdown(f"""
        As a CFA Charterholder and Investment Professional at **"Apex Financial Group"**, your role extends beyond traditional portfolio management to ensuring the ethical and compliant use of Artificial Intelligence (AI) models.
        In today's dynamic financial landscape, AI models are increasingly deployed for critical tasks like credit scoring, algorithmic trading, and sentiment analysis. However, with their power comes significant responsibility.
        The **AI Governance Committee** at Apex Financial Group, of which you are a key member, meets monthly to rigorously monitor these models.
    """)
    st.markdown(f"""
        The primary goal is to ensure that AI models operate within established risk and fairness parameters, adhere to regulatory standards (like SR 11-7 for ongoing monitoring),
        and maintain their integrity over time. Without effective oversight, models can silently degrade, leading to financial losses, reputational damage, and regulatory penalties.
    """)
    st.markdown(f"""
        This dashboard provides a crucial real-world workflow: accessing and analyzing a multi-panel monitoring dashboard, identifying deviations, and generating a governance report.
        This process transforms raw monitoring data into actionable governance intelligence, providing an "early warning system" for your committee's "cockpit."
        You'll be performing tasks that directly support **CFA Standard V(A): Diligence and Reasonable Basis**, by ensuring continuous scrutiny of AI models.
    """)

    st.subheader("Simulated Monitoring Data")
    st.markdown(f"""
        To simulate a real-world scenario for the AI Governance Committee, we generated a synthetic dataset representing 12 months of monitoring metrics for six distinct AI models used by Apex Financial Group.
        This data includes various performance indicators, fairness metrics, and operational statistics, with realistic noise, seasonal variations, and critically, injected deterioration events to test the effectiveness of our monitoring system.
    """)
    st.markdown(f"**Injected Events for Testing Governance:**")
    st.markdown(f"""
        *   **Performance Deterioration:** The 'Trading RL Agent' experiences a significant drop in `auc` in month 8.
        *   **Fairness Deterioration:** The 'Credit XGBoost' model shows a drop in `dir` in month 9.
    """)
    st.markdown(f"This allows the committee to practice identifying and responding to real-world challenges, mimicking the 'early warning system' role of continuous monitoring required by financial regulations.")

    st.subheader("AI Governance Traffic-Light Status Engine")
    st.markdown(f"""
        For the AI Governance Committee, quickly understanding the health of each model is paramount. Reviewing raw metrics for six models over 12 months is inefficient.
        As a CFA, you need a concise, actionable summary. This is where the "Traffic-Light Status Engine" comes in. It translates complex metrics into an easily digestible 'GREEN', 'YELLOW', or 'RED' status, akin to a control panel in a cockpit.
    """)
    st.markdown(f"""
        This engine implements the governance framework's thresholds across multiple dimensions: performance (AUC), fairness (Disparate Impact Ratio), data drift (PSI), human oversight (override rate), operational efficiency (unresolved alerts), and regulatory compliance (validation currency).
        The core principle for aggregation is **conservatism**: the worst individual flag determines the overall model status, ensuring that any potential issue immediately draws the committee's attention. This aligns with financial risk management principles where potential downsides are prioritized.
    """)
    st.markdown(f"**Traffic-Light Thresholds:**")
    st.markdown("""
        | Metric               | GREEN                                 | YELLOW                                 | RED                                   |
        | :------------------- | :------------------------------------ | :------------------------------------- | :------------------------------------ |
        | **AUC**              | $\ge 0.75$                            | $0.70 - 0.75$                          | $< 0.70$                              |
        | **Disparate Impact Ratio** | $\ge 0.80$                            | $0.75 - 0.80$                          | $< 0.75$                              |
        | **PSI (drift)**      | $< 0.15$                              | $0.15 - 0.25$                          | $> 0.25$                              |
        | **Override Rate**    | $< 15\%$                              | $15\% - 25\%$                          | $> 25\%$                              |
        | **Unresolved Alerts**| $< 1$                                 | $2 - 3$                                | $> 3$                                 |
        | **Validation Currency**| Within cycle                          | $> 80\%$ of cycle                      | Overdue                               |
    """)
    st.markdown(f"""
        The overall status for a model is determined by the "worst" flag across all metrics (e.g., any RED flag makes the model RED overall).
        This ensures maximum vigilance and prompt attention to potential issues.
    """)

#### **2. "Dashboard" Page**

*   **Content**: Displays the interactive Plotly dashboard panels.
*   **Widgets/Markdown**:
    ```python
    st.header("Interactive AI Model Governance Dashboard")
    st.markdown(f"""
        As a CFA, understanding the historical trends of key metrics is crucial for identifying patterns, validating the effectiveness of interventions, and predicting potential future issues.
        This dashboard presents a multi-panel view, translating raw monitoring data into clear, actionable insights for the AI Governance Committee.
    """)

    st.subheader("1. Model Performance (AUC) and Fairness (DIR) Over Time")
    st.markdown(f"""
        Understanding the historical trends of key metrics is crucial for identifying patterns, validating the effectiveness of interventions, and predicting potential future issues.
        Declining AUC could mean financial underperformance, while a deteriorating DIR raises ethical and legal concerns. You, as a CFA, need to be able to spot these trends immediately.
    """)
    st.markdown(f"**Concepts:**")
    st.markdown(f"**Area Under the Curve (AUC):** Measures a model's ability to distinguish between classes (e.g., good vs. bad credit risks). A higher AUC indicates better performance.")
    st.markdown(r"$$ AUC \in [0,1] $$") # Formula
    st.markdown(f"where $AUC$ is the Area Under the Receiver Operating Characteristic curve, a common metric for classification model performance.")
    
    st.markdown(r"**Disparate Impact Ratio (DIR):** Quantifies fairness by comparing the selection rates of different demographic groups. For example, if $P(Y=1|D=unprivileged)$ is the probability of a positive outcome for the unprivileged group and $P(Y=1|D=privileged)$ for the privileged group, then:")
    st.markdown(r"$$ DIR = \frac{{P(Y=1|D=unprivileged)}}{{P(Y=1|D=privileged)}} $$") # Formula
    st.markdown(r"where $P(Y=1|D=unprivileged)$ is the probability of a positive outcome for the unprivileged group and $P(Y=1|D=privileged)$ is the probability of a positive outcome for the privileged group. A $DIR$ close to 1 indicates fairness, while values significantly below 1 (e.g., $<0.8$) suggest disparate impact against the unprivileged group.")
    
    # Function invocation: Assumes build_performance_fairness_dashboard returns a Plotly figure.
    perf_fair_fig = build_performance_fairness_dashboard(st.session_state.monitoring_df)
    st.plotly_chart(perf_fair_fig, use_container_width=True)
    st.markdown(f"""
        **Insight**: For example, a CFA would immediately notice the sharp drop in AUC for the 'Trading RL Agent' around month 8, crossing the yellow and potentially red thresholds.
        Similarly, the 'Credit XGBoost' DIR plot might show a decline in month 9, indicating a potential fairness issue.
        These visual cues are invaluable for the AI Governance Committee to quickly identify models requiring deeper investigation, allowing for proactive risk mitigation.
    """)

    st.subheader("2. Data Drift (PSI), Human Overrides, and Alert Trends")
    st.markdown(f"""
        Beyond direct performance and fairness, the health of an AI model is also dictated by the stability of its input data, the level of human trust (or distrust), and the efficiency of operational support.
        For a CFA, data drift can signal a fundamental change in market conditions or customer behavior, invalidating the model's original training assumptions. High human override rates could indicate the model is no longer fit-for-purpose or that its recommendations are not trusted. An increasing backlog of unresolved alerts suggests operational inefficiencies or underlying systemic issues.
    """)
    st.markdown(f"**Concepts:**")
    st.markdown(r"**Population Stability Index (PSI):** Measures how much a population (e.g., input data distribution) has shifted over time. A low $PSI$ (e.g., $<0.1$) suggests stability, while a high $PSI$ ($>0.25$) indicates a significant shift that could degrade model performance.")
    st.markdown(r"$$ PSI = \sum_{{i=1}}^{{n}} (\text{{Actual}}_i - \text{{Expected}}_i) \ln \left( \frac{{\text{{Actual}}_i}}{{\text{{Expected}}_i}} \right) $$") # Formula
    st.markdown(r"where $\text{{Actual}}_i$ and $\text{{Expected}}_i$ are the percentages of records in bin $i$ for the current and baseline periods, respectively. $n$ is the number of bins.")

    st.markdown(f"**Human Override Rate:** The proportion of model-generated decisions that are manually altered by human operators. A rising rate is an 'underappreciated signal' that may indicate model degradation or miscalibration, even before performance metrics like AUC detect it.")
    st.markdown(f"**Alert Trends:** Tracking total alerts and, more critically, unresolved alerts helps assess the operational burden and responsiveness of the model support team.")

    # Function invocation: Assumes build_drift_override_alerts_dashboard returns a Plotly figure.
    drift_override_alerts_fig = build_drift_override_alerts_dashboard(st.session_state.monitoring_df)
    st.plotly_chart(drift_override_alerts_fig, use_container_width=True)
    st.markdown(f"""
        **Insight**: A CFA would use these to identify:
        *   **PSI spikes:** For instance, if 'Credit XGBoost' shows a sustained increase in PSI, it suggests that the characteristics of new credit applicants might have changed significantly, impacting the model's predictive power.
        *   **Rising override rates:** A consistently high or increasing human override rate for 'Trading RL Agent' could indicate that human traders are losing trust in the model's recommendations, potentially due to market shifts the model hasn't adapted to. This is an early warning signal, possibly preceding a drop in AUC.
        *   **Unresolved alerts:** A growing red bar for 'Unresolved Alerts' might indicate that the model support team is overwhelmed or that issues are not being addressed promptly, posing operational risks.
    """)

    st.subheader("3. Latest AI Model Inventory Status (Traffic-Light Summary)")
    st.markdown(f"""
        The AI Governance Committee needs a high-level overview of the entire AI model inventory's health at a glance. After reviewing detailed trends, the most critical piece of information is the current "traffic-light" status for each model.
        This summary acts as the final "cockpit" display, providing immediate actionable intelligence: which models are GREEN (healthy), YELLOW (under investigation), or RED (requiring immediate action).
        For a CFA, this directly informs the committee's agenda, allowing them to allocate discussion time and resources to the most pressing issues. This visualization quickly answers the question: "Are all models accounted for and within their review cycle, and what is their current health?"
    """)
    # Function invocation: Assumes build_inventory_status_dashboard returns a Plotly figure.
    inventory_status_fig = build_inventory_status_dashboard(st.session_state.monitoring_df)
    st.plotly_chart(inventory_status_fig, use_container_width=True)
    st.markdown(f"""
        **Insight**: A CFA can immediately see how many models are green, yellow, or red. This visual summary is crucial for the AI Governance Committee's monthly meeting. For example, if two models are flagged 'RED', the committee's agenda will immediately prioritize these models,
        triggering incident response protocols (e.g., "freeze model, investigate within 48h, report to CRO").
        This dashboard effectively transforms complex monitoring data into clear, actionable governance intelligence, aligning perfectly with the committee's need for a concise "cockpit" view.
    """)

#### **3. "Governance Report" Page**

*   **Content**: Generates and displays a text-based monthly governance report with executive summary and action items.
*   **Widgets/Markdown**:
    ```python
    st.header("Monthly AI Governance Report")
    st.markdown(f"""
        The AI Governance Committee's review culminates in a formal monthly report. As a CFA, your responsibility includes not only identifying issues but also documenting them,
        summarizing findings for executive leadership, and explicitly defining action items. A dashboard is an excellent monitoring tool, but a structured report is the formal record for compliance, audits, and strategic decision-making.
    """)
    st.markdown(f"""
        This report provides an executive summary, clearly stating the overall health of the model inventory (counts of Green/Yellow/Red models), and crucially, details specific action items for all non-green models.
        This ensures transparency, accountability, and a clear path for remediation, fulfilling regulatory requirements for ongoing monitoring and incident management.
    """)

    # Function invocation: Capture print output from generate_monthly_report
    f = io.StringIO()
    with redirect_stdout(f):
        generate_monthly_report(st.session_state.monitoring_df)
    report_output = f.getvalue()
    st.text(report_output) # Render the captured text output

    st.markdown(f"""
        **Insight**: The generated text report provides a clear, concise summary of the AI model inventory's status for the most recent month.
        It highlights the total number of models in each traffic-light category and, crucially, lists specific action items for any models flagged as 'YELLOW' or 'RED'.
        For instance, seeing "Trading RL Agent" as RED with an "AUC below 0.70" flag, followed by the action "Freeze model. Investigate within 48h, Report to CRO," provides the clear directive a CFA needs to initiate the incident response protocol.
        This report is the official output of the AI Governance Committee's monthly review, ensuring accountability and driving timely corrective measures, a core aspect of responsible AI governance in finance. It directly supports regulatory compliance by providing documented evidence of ongoing oversight and action.
    """)
    ```

