import streamlit as st
import pandas as pd
from datetime import datetime
import io
from contextlib import redirect_stdout

# Source (simulation + traffic-light rules)
from source import simulate_monitoring_data, compute_traffic_light, generate_monthly_report

import plotly.express as px

st.set_page_config(
    page_title="QuLab: Lab 47: Monitoring Dashboard Example", layout="wide")

# -----------------------------
# Sidebar + Header
# -----------------------------
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()

st.title("QuLab: Lab 47: Monitoring Dashboard Example")
st.caption("A training-grade monitoring & governance workflow for CFA charterholders and investment professionals.")
st.divider()

# -----------------------------
# Session State
# -----------------------------


def _initialize_state():
    if "seed" not in st.session_state:
        st.session_state.seed = 7

    if "n_months" not in st.session_state:
        st.session_state.n_months = 12

    if "monitoring_df" not in st.session_state:
        df = simulate_monitoring_data(
            n_months=st.session_state.n_months, seed=st.session_state.seed)
        df[["status", "flags"]] = df.apply(
            lambda row: pd.Series(compute_traffic_light(row)), axis=1)

        # A conservative "primary driver" label for one-glance triage
        def primary_flag(flags):
            if not isinstance(flags, list) or len(flags) == 0:
                return "No threshold breaches"
            priority = ["AUC", "DIR", "PSI", "Overrides",
                        "Unresolved Alerts", "Validation Currency"]
            for p in priority:
                for f in flags:
                    if p.lower() in str(f).lower():
                        return str(f)
            return str(flags[0])

        df["primary_flag"] = df["flags"].apply(primary_flag)

        st.session_state.monitoring_df = df

    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    if "guided_mode" not in st.session_state:
        st.session_state.guided_mode = True


_initialize_state()

df_all = st.session_state.monitoring_df.copy()
months = sorted(df_all["month"].dropna().unique())

# -----------------------------
# Sidebar Controls (non-technical framing)
# -----------------------------
st.sidebar.title("Workflow (3 steps)")
options = ["Overview", "Dashboard", "Governance Report"]
current_index = options.index(
    st.session_state.page) if st.session_state.page in options else 0
page_selection = st.sidebar.selectbox("Step", options, index=current_index)

if page_selection != st.session_state.page:
    st.session_state.page = page_selection
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Committee meeting (as-of)")
selected_month = st.sidebar.selectbox(
    "Select the month you are reviewing",
    months,
    index=len(months) - 1,
    format_func=lambda d: pd.to_datetime(d).strftime("%B %Y"),
    help="Monitoring and governance are time-indexed. Pick the committee meeting month you want to evaluate.",
)

st.sidebar.checkbox(
    "Guided learning mode",
    value=st.session_state.guided_mode,
    help="Adds quick checkpoints, watch-outs, and decision prompts while you review the dashboard.",
    key="guided_mode",
)

with st.sidebar.expander("Simulation controls (training)", expanded=False):
    st.number_input("Months (simulated)", min_value=6,
                    max_value=36, step=1, key="n_months")
    st.number_input("Random seed", min_value=0,
                    max_value=10_000, step=1, key="seed")
    if st.button("Regenerate simulated data"):
        df = simulate_monitoring_data(
            n_months=st.session_state.n_months, seed=st.session_state.seed)
        df[["status", "flags"]] = df.apply(
            lambda row: pd.Series(compute_traffic_light(row)), axis=1)

        def primary_flag(flags):
            if not isinstance(flags, list) or len(flags) == 0:
                return "No threshold breaches"
            priority = ["AUC", "DIR", "PSI", "Overrides",
                        "Unresolved Alerts", "Validation Currency"]
            for p in priority:
                for f in flags:
                    if p.lower() in str(f).lower():
                        return str(f)
            return str(flags[0])

        df["primary_flag"] = df["flags"].apply(primary_flag)
        st.session_state.monitoring_df = df
        st.rerun()

# -----------------------------
# Global provenance banner (numbers must have meaning)
# -----------------------------
sim_through = pd.to_datetime(df_all["month"].max()).strftime("%B %Y")
review_month_str = pd.to_datetime(selected_month).strftime("%B %Y")
st.markdown(
    f"**App refreshed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
    f"**Data type:** Synthetic (training) • **Simulated through:** {sim_through} • **Committee month selected:** {review_month_str}"
)

st.info(
    "Training constraint: treat every number as an input to a governance decision. "
    "You should be able to explain (i) what it measures, (ii) how it is computed, (iii) what threshold it is judged against, "
    "and (iv) what action it triggers when breached."
)

# Filter view for "as-of" review month
df_upto = df_all[df_all["month"] <= selected_month].copy()
df_month = df_all[df_all["month"] == selected_month].copy()

# Ensure primary_flag exists (defensive programming for session state issues)
if "primary_flag" not in df_all.columns:
    def primary_flag(flags):
        if not isinstance(flags, list) or len(flags) == 0:
            return "No threshold breaches"
        priority = ["AUC", "DIR", "PSI", "Overrides",
                    "Unresolved Alerts", "Validation Currency"]
        for p in priority:
            for f in flags:
                if p.lower() in str(f).lower():
                    return str(f)
        return str(flags[0])

    df_all["primary_flag"] = df_all["flags"].apply(primary_flag)
    df_upto["primary_flag"] = df_upto["flags"].apply(primary_flag)
    df_month["primary_flag"] = df_month["flags"].apply(primary_flag)
    st.session_state.monitoring_df["primary_flag"] = st.session_state.monitoring_df["flags"].apply(
        primary_flag)


# -----------------------------
# Helpers (charts)
# -----------------------------
def _add_threshold_lines(fig, thresholds):
    # thresholds: list of tuples (y, label)
    for y, label in thresholds:
        fig.add_hline(y=y, line_dash="dash", annotation_text=label,
                      annotation_position="top left")
    return fig


def _add_asof_vline(fig, asof_month):
    fig.add_vline(x=asof_month, line_dash="dot")
    return fig


def _status_color_map():
    return {"GREEN": "green", "YELLOW": "gold", "RED": "red"}


def _model_order(df):
    return list(df["model"].drop_duplicates())


# -----------------------------
# Page: Overview
# -----------------------------
if st.session_state.page == "Overview":
    st.header("What this lab trains: monthly model-risk monitoring decisions")

    st.markdown(
        """
As a CFA Charterholder and Investment Professional at **"Apex Financial Group"**, your role extends beyond traditional portfolio management to ensuring the ethical and compliant use of Artificial Intelligence (AI) models.
In today's dynamic financial landscape, AI models are increasingly deployed for critical tasks like credit scoring, algorithmic trading, and sentiment analysis. However, with their power comes significant responsibility.

This app is designed as a **monthly AI governance workflow**:
1) Review monitoring evidence, 2) tie signals to explicit thresholds, and 3) document audit-ready actions.
"""
    )

    colA, colB = st.columns([1.15, 1])
    with colA:
        st.subheader("Start here (3-step checklist)")
        st.markdown(
            """
- **Step 1 (Overview):** Understand *what each metric means* and *what thresholds imply*.
- **Step 2 (Dashboard):** Identify which models breach which thresholds **in the selected month**.
- **Step 3 (Governance Report):** Convert those breaches into **documented actions** (owner + urgency + interim control).
"""
        )
    with colB:
        st.subheader("Policy mapping (how a committee should react)")
        st.markdown(
            """
- **GREEN:** Continue normal operations; monitor at the standard cadence.
- **YELLOW:** Investigate + document owner and timeline; verify whether deterioration is persistent.
- **RED:** Immediate controls (freeze/limit), escalation, and remediation plan; treat as a governance breach.
"""
        )

    st.divider()
    st.subheader("Injected events for training (so you know what to detect)")
    st.markdown("**Injected Events for Testing Governance:**")
    st.markdown(
        """
1) The **Trading RL Agent** experiences a significant drop in **AUC** in month 8 (performance deterioration).
2) The **Credit XGBoost** model shows a decline in **DIR** in month 9 (fairness risk).
3) The **FinBERT Sentiment** model experiences a gradual increase in **PSI** starting in month 5 (data drift).
4) The **Research Copilot** has increasing **human override rates** from month 6 onwards (loss of user trust).
5) The **Rebalancing Agent** accumulates increasing **unresolved alerts** from month 7 onwards (operational backlog).
6) The **ESG Agent** becomes overdue for **validation** in month 10 (validation currency breach).
"""
    )
    st.caption(
        "Training note: some series are intentionally stressed to practice governance response logic. "
        "Your job is not to admire the line chart—it's to map signals → thresholds → actions."
    )

    st.divider()
    st.subheader(
        "Traffic-Light Status Engine (explicit thresholds + conservative aggregation)")
    st.markdown(
        """
These thresholds are a training proxy for the kinds of **policy limits** used in financial model governance.
**Aggregation rule:** the model’s overall status is the **worst** flag across dimensions (a conservative committee posture).
"""
    )

    st.markdown(
        """
| Metric | GREEN | YELLOW | RED |
|---|---:|---:|---:|
| AUC | ≥ 0.75 | 0.70–0.75 | < 0.70 |
| DIR | ≥ 0.90 | 0.80–0.90 | < 0.80 |
| PSI | < 0.10 | 0.10–0.25 | > 0.25 |
| Override Rate | < 5% | 5%–10% | > 10% |
| Unresolved Alerts | < 5 | 5–10 | > 10 |
| Validation Currency | Within cycle | Approaching overdue | Overdue |
"""
    )

    st.caption(
        "Threshold rationale (training): YELLOW = early warning; RED = breach requiring immediate controls. "
        "In production, thresholds are calibrated to model materiality, error costs, and supervisory expectations."
    )

    with st.expander("Tier rule: validation currency depends on materiality (Tier)", expanded=False):
        st.markdown(
            """
- **Tier 1:** must be revalidated at least annually (≈ 12 months / 365 days)  
- **Tier 2:** at least every 24 months (≈ 730 days)  
- **Tier 3:** at least every 36 months (≈ 1095 days)
"""
        )

    st.divider()
    st.subheader("Key monitoring concepts (definitions + formulas)")

    st.markdown(f"**Concepts:**")
    st.markdown(
        f"**Area Under the Curve (AUC):** Measures a model's ability to distinguish between classes (e.g., good vs. bad credit risks). A higher AUC indicates better performance."
    )
    st.markdown(r"$$ AUC \in [0,1] $$")
    st.markdown(
        f"where $AUC$ is the Area Under the Receiver Operating Characteristic curve, a common metric for classification model performance."
    )

    st.markdown(
        f"**Disparate Impact Ratio (DIR):** Quantifies fairness by comparing the selection rates of different demographic groups. For example, if $P(Y=1|D=unprivileged)$ is the probability of a positive outcome for the unprivileged group and $P(Y=1|D=privileged)$ for the privileged group, then:"
    )
    st.markdown(
        r"$$ DIR = \frac{P(Y=1|D=unprivileged)}{P(Y=1|D=privileged)} $$")
    st.markdown(
        r"where $P(Y=1|D=unprivileged)$ is the probability of a positive outcome for the unprivileged group and $P(Y=1|D=privileged)$ is the probability of a positive outcome for the privileged group. A $DIR$ close to 1 indicates fairness, while values significantly below 1 (e.g., $<0.8$) suggest disparate impact against the unprivileged group."
    )

    st.markdown(
        r"**Population Stability Index (PSI):** Measures how much a population (e.g., input data distribution) has shifted over time. A low $PSI$ (e.g., $<0.1$) suggests stability, while a high $PSI$ ($>0.25$) indicates a significant shift that could degrade model performance."
    )
    st.markdown(
        r"$$ PSI = \sum_{i=1}^{n} (\text{Actual}_i - \text{Expected}_i) \ln \left( \frac{\text{Actual}_i}{\text{Expected}_i} \right) $$"
    )
    st.markdown(
        r"where $\text{Actual}_i$ and $\text{Expected}_i$ are the percentages of records in bin $i$ for the current and baseline periods, respectively. $n$ is the number of bins."
    )

    st.markdown(
        f"**Human Override Rate:** The proportion of model-generated decisions that are manually altered by human operators. A rising rate is an 'underappreciated signal' that may indicate model degradation or miscalibration, even before performance metrics like AUC detect it."
    )
    st.markdown(
        f"**Alert Trends:** Tracking total alerts and, more critically, unresolved alerts helps assess the operational burden and responsiveness of the model support team."
    )

    if st.session_state.guided_mode:
        st.success(
            "Checkpoint: You should now be able to answer: "
            "Which metrics measure (i) predictive ranking quality, (ii) fairness, (iii) data drift, (iv) human trust, and (v) operational control health?"
        )
        with st.expander("Common misconceptions (watch-outs)", expanded=False):
            st.markdown(
                """
- **High PSI does not automatically mean the model is wrong.** It means the input population shifted; you still need to evaluate performance impact.
- **Stable AUC does not imply stable fairness.** Fairness can deteriorate even when ranking performance looks unchanged.
- **Override rate can increase due to policy changes.** Confirm no policy/process change before attributing it to model decay.
- **Alert counts are not automatically model risk.** They indicate control burden; risk depends on alert type and SLA breaches.
"""
            )

# -----------------------------
# Page: Dashboard
# -----------------------------
elif st.session_state.page == "Dashboard":
    st.header("Monitoring dashboard (evidence for the committee)")

    st.markdown(
        """
This page is designed to support a governance meeting. Read each panel as:
**(1) trend**, **(2) threshold breach**, **(3) plausible cause**, and **(4) required action**.
"""
    )

    # ---- Panel 1: Performance + Fairness
    st.subheader(
        "1) Performance & Fairness: are we still accurate, and are we still equitable?")
    st.caption(
        "Interpretation discipline: AUC is ranking quality; DIR is selection-rate parity. "
        "Both require a clear outcome definition and group definitions to be meaningful."
    )

    # AUC
    auc_df = df_upto[df_upto["auc"].notna()].copy()
    fig_auc = px.line(
        auc_df, x="month", y="auc", color="model", markers=True, title="Model Performance (AUC) Over Time"
    )
    fig_auc = _add_threshold_lines(
        fig_auc,
        [(0.75, "AUC GREEN (≥ 0.75)"), (0.70, "AUC YELLOW/RED boundary (0.70)")],
    )
    fig_auc = _add_asof_vline(fig_auc, selected_month)
    fig_auc.update_yaxes(range=[0.5, 0.95], title_text="AUC")
    st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown(
        """
**Decision translation:**
- If AUC crosses **YELLOW**: commission root-cause analysis and document an owner + timeline.
- If AUC crosses **RED**: consider immediate controls (freeze/limit) and escalate based on model tier/materiality.
"""
    )

    # DIR
    dir_df = df_upto[df_upto["dir"].notna()].copy()
    fig_dir = px.line(
        dir_df, x="month", y="dir", color="model", markers=True, title="Model Fairness (Disparate Impact Ratio) Over Time"
    )
    fig_dir = _add_threshold_lines(
        fig_dir, [(0.90, "DIR GREEN (≥ 0.90)"), (0.80, "DIR RED boundary (0.80)")])
    fig_dir = _add_asof_vline(fig_dir, selected_month)
    fig_dir.update_yaxes(range=[0.5, 1.0], title_text="DIR")
    st.plotly_chart(fig_dir, use_container_width=True)

    st.markdown(
        """
**Decision translation:**
- If DIR crosses **YELLOW/RED**: trigger fairness review (confirm group definitions + decision definition, perform adverse impact analysis, review policy constraints).
"""
    )

    if st.session_state.guided_mode:
        st.divider()
        st.subheader("Checkpoint (guided)")

        month_view = df_month.copy()
        auc_month = month_view[month_view["auc"].notna()].copy()
        auc_red_models = auc_month[auc_month["auc"] < 0.70]["model"].tolist()

        st.markdown(
            "**Q1:** In the selected month, which model(s) breach **AUC RED** (< 0.70)?")
        guess_auc_red = st.multiselect(
            "Select model(s)", _model_order(df_all), key="guess_auc_red")
        if st.button("Check Q1", key="check_q1"):
            if set(guess_auc_red) == set(auc_red_models):
                st.success(
                    f"Correct. AUC RED model(s): {', '.join(auc_red_models) if auc_red_models else 'None'}.")
            else:
                st.info(
                    f"Not quite. AUC RED model(s) in {review_month_str}: "
                    f"{', '.join(auc_red_models) if auc_red_models else 'None'}."
                )

        st.markdown(
            "**Q2:** In the selected month, which model(s) breach **DIR RED** (< 0.80)?")
        dir_month = month_view[month_view["dir"].notna()].copy()
        dir_red_models = dir_month[dir_month["dir"] < 0.80]["model"].tolist()
        guess_dir_red = st.multiselect(
            "Select model(s)", _model_order(df_all), key="guess_dir_red")
        if st.button("Check Q2", key="check_q2"):
            if set(guess_dir_red) == set(dir_red_models):
                st.success(
                    f"Correct. DIR RED model(s): {', '.join(dir_red_models) if dir_red_models else 'None'}.")
            else:
                st.info(
                    f"Not quite. DIR RED model(s) in {review_month_str}: "
                    f"{', '.join(dir_red_models) if dir_red_models else 'None'}."
                )

    st.divider()

    # ---- Panel 2: Drift + Overrides + Alerts
    st.subheader(
        "2) Diagnostics: is it data shift, human override, or control backlog?")
    st.caption(
        "PSI is a distribution-shift indicator (baseline vs current). Override rate is a human trust signal. "
        "Unresolved alerts indicate control backlog and SLA pressure."
    )

    # PSI
    fig_psi = px.line(df_upto, x="month", y="psi", color="model",
                      markers=True, title="Data Drift (PSI) Over Time")
    fig_psi = _add_threshold_lines(
        fig_psi, [(0.10, "PSI YELLOW (0.10)"), (0.25, "PSI RED (0.25)")])
    fig_psi = _add_asof_vline(fig_psi, selected_month)
    fig_psi.update_yaxes(range=[0.0, 0.45], title_text="PSI")
    st.plotly_chart(fig_psi, use_container_width=True)

    st.markdown(
        """
**Decision translation:**
- PSI rises into **YELLOW**: increase monitoring attention; check whether drift is benign vs performance-relevant.
- PSI crosses **RED**: investigate data pipeline changes, market regime shifts, and model stability assumptions.
"""
    )

    # Override rate
    fig_ovr = px.line(
        df_upto, x="month", y="override_rate", color="model", markers=True, title="Human Override Rate Over Time"
    )
    fig_ovr = _add_threshold_lines(
        fig_ovr, [(0.05, "Override YELLOW (5%)"), (0.10, "Override RED (10%)")])
    fig_ovr = _add_asof_vline(fig_ovr, selected_month)
    fig_ovr.update_yaxes(range=[0.0, 0.25], title_text="Override rate")
    st.plotly_chart(fig_ovr, use_container_width=True)

    st.markdown(
        """
**Decision translation:**
- Rising overrides: validate whether the model’s recommendation set is still aligned with policy and market reality.
- Overrides can lead performance deterioration: treat as an early warning signal, not a footnote.
"""
    )

    # Unresolved alerts
    fig_alerts = px.bar(
        df_upto,
        x="month",
        y="unresolved_alerts",
        color="model",
        title="Unresolved Alerts Over Time (control backlog)",
        barmode="group",
    )
    fig_alerts = _add_threshold_lines(
        fig_alerts, [(5, "Alerts YELLOW (5)"), (10, "Alerts RED (10)")])
    fig_alerts = _add_asof_vline(fig_alerts, selected_month)
    fig_alerts.update_yaxes(range=[0, max(12, int(
        df_upto["unresolved_alerts"].max()) + 2)], title_text="Unresolved alerts")
    st.plotly_chart(fig_alerts, use_container_width=True)

    st.markdown(
        """
**Decision translation:**
- If unresolved alerts move into **YELLOW/RED**: treat as control effectiveness risk; escalate resourcing and SLA adherence.
"""
    )

    if st.session_state.guided_mode:
        with st.expander("Watch-outs (guided)", expanded=False):
            st.markdown(
                """
- PSI requires a defined baseline period (“Expected”) and a current window (“Actual”). If those move, PSI interpretation changes.
- Overrides may rise after **policy changes**, not just model decay. Confirm any policy/process changes first.
- Alerts should be interpreted only after clarifying alert taxonomy (monitoring breach vs data incident vs workflow exception).
"""
            )

    st.divider()

    # ---- Panel 3: Inventory status (as-of month)
    st.subheader(
        "3) Inventory (selected month): which models need committee time now?")
    st.caption(
        "This panel is categorical: color indicates governance status; it is not a magnitude chart.")

    inv = df_month.copy()
    fig_inv = px.bar(
        inv.sort_values(["status", "tier"]),
        x="model",
        y=[1] * len(inv),
        color="status",
        color_discrete_map=_status_color_map(),
        title=f"Model Inventory Status (as-of {review_month_str})",
        hover_data={"primary_flag": True, "tier": True,
                    "model": True, "status": True},
    )
    fig_inv.update_yaxes(visible=False)
    fig_inv.update_xaxes(title_text="Model")
    st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown(
        """
**Decision translation:**
- Any **RED** model must appear on the committee agenda with an immediate control and an escalation path.
- Multiple **YELLOW** models should be prioritized by tier/materiality and persistence across months.
"""
    )

    st.subheader("Selected-month triage table (status + primary driver)")
    triage_cols = [
        "model",
        "tier",
        "status",
        "primary_flag",
        "auc",
        "dir",
        "psi",
        "override_rate",
        "unresolved_alerts",
        "last_validation",
    ]
    st.dataframe(inv[triage_cols].sort_values(["status", "tier"]),
                 use_container_width=True, hide_index=True)

# -----------------------------
# Page: Governance Report
# -----------------------------
elif st.session_state.page == "Governance Report":
    st.header("Monthly governance minutes (audit-ready output)")

    st.markdown(
        """
A dashboard helps you detect issues. A governance report is the formal record:
**breach reason → action → owner → urgency**.
Use this page to translate monitoring evidence into committee minutes suitable for oversight.
"""
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("As-of month")
        st.write(f"**{review_month_str}**")
        st.caption(
            "The report is generated for the selected committee month (not necessarily the latest simulated month).")
    with col2:
        st.subheader("What a good report must contain")
        st.markdown(
            """
- Clear status totals (GREEN/YELLOW/RED)
- Model-by-model breaches with the **specific trigger**
- Actions that match policy (investigate vs immediate controls)
- A statement of assumptions and limitations
"""
        )

    st.divider()

    st.subheader("Assumptions & limitations (must be explicit)")
    st.warning(
        "This lab uses synthetic data for training purposes. Thresholds and actions are illustrative. "
        "In production governance, limits are calibrated to model materiality, error costs, and regulatory expectations."
    )

    # Generate the text-based report from the source module (captures stdout)
    buf = io.StringIO()
    with redirect_stdout(buf):
        generate_monthly_report(df_all, report_month=selected_month)
    report_text = buf.getvalue()

    st.subheader("Committee minutes (generated)")
    st.code(report_text, language="text")

    if st.session_state.guided_mode:
        st.subheader("Decision prompts (guided)")
        red_models = df_month[df_month["status"] == "RED"]["model"].tolist()
        yellow_models = df_month[df_month["status"]
                                 == "YELLOW"]["model"].tolist()

        st.markdown(
            f"""
- **In {review_month_str}:** RED model(s) = **{', '.join(red_models) if red_models else 'None'}**; YELLOW model(s) = **{', '.join(yellow_models) if yellow_models else 'None'}**.
- For each non-green model: state the breach driver and write one interim control you would impose until remediation completes.
"""
        )

    st.divider()
    st.subheader("Download")
    st.download_button(
        "Download report (text)",
        data=report_text,
        file_name=f"ai_governance_report_{pd.to_datetime(selected_month).strftime('%Y_%m')}.txt",
        mime="text/plain",
    )


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
