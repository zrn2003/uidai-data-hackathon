# Aadhaar Sentinel 🛡️

**An AI-Powered Integrity & Pattern Recognition System**

Aadhaar Sentinel is an intelligent command center designed to identify anomalies, forecast demand, and provide actionable insights into Aadhaar enrolment and update operations. It goes beyond simple dashboards by acting as a proactive "Watchdog".

## 🚀 Key Capabilities

### 1. 🧠 Explainable Anomaly Detection (XAI)
Instead of a "Black Box" alert, Sentinel tells you **why**:
*   *Before*: "Anomaly Score: -0.45"
*   *Sentinel*: "⚠️ **Suspicious Surge**: This Pincode processed **2,500 biometric updates**, which is **450% higher** than the district's typical daily volume."
*   **Methodology**: Uses `IsolationForest` for unsupervised outlier detection, coupled with a Z-Score based diagnostic layer for explanation.

### 2. 🌍 Hierarchical Intelligence (State to Street)
Full drill-down capability:
1.  **National View**: Heatmap of all states.
2.  **District View**: Identify problem areas within a state.
3.  **Pincode View**: Pinpoint the exact locality (and potentially operator center) causing the anomaly.

### 3. 🛡️ Data Fusion
Integrates and normalizes three distinct streams:
*   **Enrolment Data**
*   **Demographic Updates** (Name, Address, etc.)
*   **Biometric Updates** (Fingerprint, Iris)

## 🛠️ Tech Stack
*   **Core**: Python 3.9+
*   **Engine**: Pandas (ETL), Scikit-Learn (Anomaly Detection)
*   **Interface**: Streamlit (Reactive Dashboard)
*   **Visualization**: Plotly Express (Interactive Charts)

## 📦 Setup & Usage

### Prerequisites
*   Python 3.8 or higher.
*   The `dataset/` folder containing data CSVs.

### Installation
```bash
git clone https://github.com/zrn2003/uidai-data-hackathon.git
cd uidai-data-hackathon
pip install -r requirements.txt
```

### Running the Sentinel
```bash
python -m streamlit run src/app.py
```

## 📂 Project Structure
*   `dataset/`: Raw CSV Data.
*   `src/data_loader.py`: Ingestion & "Golden Record" merging logic.
*   `src/models.py`: ML logic (Isolation Forest + Explanation Engine).
*   `src/app.py`: The Main Command Center Application.

---
*Submitted for the UIDAI Data Hackathon 2024*
