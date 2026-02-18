# QuLab: Lab 47: AI Model Governance Monitoring Dashboard Example

![Quant University Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Lab 47: AI Model Governance Monitoring Dashboard Example** is a Streamlit application designed for Investment Professionals and CFA Charterholders within financial institutions. It simulates a crucial real-world scenario: overseeing and governing Artificial Intelligence (AI) models used in critical financial applications such as credit scoring, algorithmic trading, and sentiment analysis.

The application functions as an "early warning system" and "cockpit" for an AI Governance Committee, enabling continuous monitoring of AI models against established risk and fairness parameters, regulatory standards (like SR 11-7), and internal policies. It transforms raw monitoring data into actionable governance intelligence, providing a clear, concise view of model health and highlighting areas requiring immediate attention. This project directly supports **CFA Standard V(A): Diligence and Reasonable Basis** by ensuring rigorous, ongoing scrutiny of AI models.

## Features

This application provides a comprehensive suite of tools for AI model governance:

*   **Interactive Multi-Panel Dashboard**: Visualize key model health metrics over time, including:
    *   **Performance (AUC)**: Track the discriminatory power of classification models.
    *   **Fairness (Disparate Impact Ratio - DIR)**: Monitor for bias and ensure equitable outcomes across demographic groups.
    *   **Data Drift (Population Stability Index - PSI)**: Detect significant shifts in input data distributions that could degrade model performance.
    *   **Human Override Rates**: Gauge the level of human trust and intervention in model decisions, an early indicator of potential issues.
    *   **Unresolved Alert Trends**: Assess operational efficiency and the responsiveness of support teams.
*   **AI Governance Traffic-Light Status Engine**: Automatically classifies the health of each AI model as `GREEN`, `YELLOW`, or `RED` based on predefined thresholds across multiple dimensions (performance, fairness, drift, operations, compliance). This "worst-flag" aggregation ensures critical issues are immediately highlighted.
*   **Simulated Deterioration Events**: The underlying dataset includes injected scenarios of performance and fairness degradation to provide realistic challenges for governance practice.
*   **Automated Monthly AI Governance Report**: Generate a formal, structured text report summarizing the overall model inventory status and listing specific action items for non-green models, crucial for compliance, audits, and executive decision-making.
*   **Contextual Information**: Detailed explanations of AI governance imperative, simulated data, metric definitions (AUC, DIR, PSI), and the traffic-light status engine logic.
*   **Responsive UI**: Built with Streamlit for an intuitive and interactive user experience.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/qu-lab-47-ai-governance.git
    cd qu-lab-47-ai-governance
    ```
    *(Note: Replace `yourusername/qu-lab-47-ai-governance` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install the required dependencies:**
    Create a `requirements.txt` file in the root of your project with the following content:
    ```
    streamlit>=1.0
    pandas
    plotly
    numpy
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is active and you are in the project's root directory.
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

2.  **Navigate the Dashboard:**
    Use the sidebar on the left to switch between the following sections:
    *   **Overview**: Learn about the AI governance imperative, the simulated data, and the logic behind the "Traffic-Light Status Engine."
    *   **Dashboard**: Interact with the multi-panel charts visualizing historical trends for model performance, fairness, data drift, human overrides, alerts, and the latest traffic-light summary.
    *   **Governance Report**: Generate a concise textual report summarizing the current state of the model inventory and outlining specific action items for models requiring attention.

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains helper functions for data simulation,
|                           # traffic light computation, dashboard plots,
|                           # and report generation.
├── requirements.txt        # List of Python dependencies
└── README.md               # This README file
```

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application user interface.
*   **Pandas**: For data manipulation and analysis.
*   **Plotly**: For generating interactive and informative data visualizations.
*   **NumPy**: For numerical operations, likely used in data simulation and metric calculations.
*   **io** and **contextlib**: For capturing standard output to display the governance report within the Streamlit app.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if applicable, otherwise state "No specific license defined for this lab project").

## Contact

For questions, feedback, or further information about Quant University's labs and programs, please visit [Quant University](https://www.quantuniversity.com/).