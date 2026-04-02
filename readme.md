
# 🧠 CrediSight – Credit Risk & Fraud Detection Platform

**End-to-End Machine Learning Web Application** for Credit Card Application Risk Assessment and Fraud Detection.

**CrediSight — Empowering Smarter, Safer Credit Decisions with AI.**

![Dashboard](screenshots/dashboard.png)  
*(Add 3-4 screenshots of the running app here)*

---

## 🚀 Business Problem
Credit card issuers lose millions due to **defaults** and **fraudulent applications**. This interactive ML solution helps analysts quickly assess risk, predict defaults, and detect suspicious applications in real-time.

---

## ✨ Key Features

- Interactive **Streamlit Dashboard** (Upload CSV or use sample data)
- Smart **Feature Engineering** (`total_delay` & `dpd_bucket`)
- Multiple ML Models in one app:
  - Application Status Classification (Approved/Rejected/Pending)
  - **Default Flag Prediction** (Gradient Boosting)
  - Processing Days Regression
  - Delay Bucket Classification (XGBoost)
  - **Fraud/Anomaly Detection** (Isolation Forest)
- Rich Visualizations (ROC Curve, Feature Importance, Heatmap, Geospatial Map, etc.)

---

## 📊 Model Performance Highlights

| Model                        | Key Metric                  | Score          |
|-----------------------------|-----------------------------|----------------|
| Default Prediction          | ROC-AUC                     | **0.84**       |
| Application Status          | Accuracy                    | **85%**        |
| Anomaly Detection           | High-Risk Flagged           | **5%**         |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Language**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, Gradient Boosting, Random Forest, Isolation Forest
- **Visualization**: Matplotlib, Seaborn

---

## 🚀 How to Run

```
git clone https://github.com/chandeswerreddychintham/CrediSight-Credit-Risk-Fraud-Detection.git
```
```
cd CrediSight-Credit-Risk-Fraud-Detection
```
```
pip install -r requirements.txt
```
```
streamlit run app.py
```
## 🧠 Project Workflow

1. **Load Data** → Upload or use built-in dataset.
2. **Preprocess Data** → Handle categorical encoding and derived features.
3. **Visualize** → Explore numeric and categorical trends.
4. **Select ML Task** → Choose prediction or anomaly detection mode.
5. **Train & Evaluate** → Run models, view metrics and plots instantly.

---


## Project Structurebash
```
📦 CrediSight
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies list
├── README.md                   # Project documentation
└── sample_data.csv             # Optional example data
```

## 🧑‍💻 Author

**Developed by:** Chandeshwer Reddy (Data Science & ML Enthusiast).
**Purpose:** Demonstration of end-to-end credit risk modeling and anomaly detection using modern ML tools.

---

## 💡 Future Enhancements

* Add **SHAP-based explainability** for feature impact.
* Integrate **live fraud alert dashboard**.
* Connect with **real-time APIs** for dynamic financial data.
* Deploy on **Streamlit Cloud / AWS / Azure** for production use.

---

**🎯 CrediSight — Empowering Smarter, Safer Credit Decisions with AI.**
