# 🏦 Credit Risk Prediction & Analytics System

> **End-to-end Credit Risk platform combining Machine Learning, SQL Analytics, Power BI Dashboards, and a Natural Language AI Agent — built to simulate a real-world bank lending intelligence system.**

---

## 📌 Project Overview

This project addresses one of the most critical problems in banking and finance — **predicting loan defaults and optimizing lending strategy**. Using a dataset of **255,347 loan applicants**, I built a complete data pipeline from raw data exploration to ML model deployment, business dashboards, and an AI-powered SQL agent that allows non-technical stakeholders to query the data in plain English.

---

## 🎯 Business Problem

> A bank needs to identify which loan applicants are likely to default before disbursing funds. Manual review of 255K+ applications is not scalable. The goal is to build an automated, explainable system that classifies risk, supports business decisions, and quantifies financial exposure.

---

## 📊 Key Business Results

| Metric | Value |
|--------|-------|
| Total Loan Portfolio | ₹32.6 Billion |
| Total Loans Analyzed | 255,347 |
| Overall Default Rate | 11.6% |
| Total Defaults Identified | 29,700 |
| Risk Exposure Flagged | ₹4.3 Billion |
| Auto Approve (Low Risk) | 99,050 loans (39%) |
| Manual Review (Medium Risk) | 106,700 loans (42%) |
| Auto Reject (High Risk) | 49,600 loans (19%) |

---

## 🏗️ Project Architecture

```
Raw Dataset (255K records)
        │
        ▼
┌───────────────────┐
│  Python — EDA &   │
│  Feature Engg.    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     ┌──────────────────────┐
│  ML Model         │────▶│  Streamlit App        │
│  (Classification) │     │  (Live Prediction)    │
└────────┬──────────┘     └──────────────────────┘
         │
         ▼
┌───────────────────┐     ┌──────────────────────┐
│  SQL Analytics    │────▶│  Power BI Dashboard   │
│  (PostgreSQL)     │     │  (Business Insights)  │
└────────┬──────────┘     └──────────────────────┘
         │
         ▼
┌───────────────────┐
│  AI SQL Agent     │
│  (n8n + Gemini)   │
│  Natural Language │
│  → SQL → Answer   │
└───────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, SHAP |
| Database | PostgreSQL (Supabase) |
| SQL Analytics | Advanced SQL (CTEs, Window Functions, NTILE, LAG/LEAD) |
| Visualization | Power BI |
| Deployment | Streamlit |
| AI Agent | n8n, Google Gemini, PostgreSQL Tool |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
Credit-Risk-Project/
│
├── 📓 credit_risk_prediction_system.ipynb   # Full EDA + Feature Engineering + ML pipeline
├── 🤖 best_credit_risk_model.pkl            # Trained ML model
├── ⚖️  scaler.pkl                            # Feature scaler
├── 🖥️  app.py                               # Streamlit web application
├── 🗄️  CREDIT_RISK_FINAL.sql               # Advanced SQL analytics (10 sections)
├── 📊 dashboard1_portfolio_overview.png
├── 📊 dashboard2_risk_driver_analysis.png
├── 📊 dashboard3_ml_model_output.png
├── 📊 dashboard4_pareto_simulation.png
├── 📊 dashboard5_business_insights.png
├── 🤖 CREDIT_RISK_SQL_AI_AGENT_WORKFLOW.json
├── 🤖 CREDIT_RISK_SQL_AI_AGENT_SUB_WORKFLOW.json
├── requirements.txt
└── README.md
```

---

## 📈 Power BI Dashboards

### 1️⃣ Loan Portfolio Performance Overview
![Dashboard 1](POWER%20BI%20DASHBOARD/dashboard1_portfolio_overview.png)
> Total disbursed ₹32.6Bn across 255K loans with 11.6% default rate. Business loans show highest expected loss at ₹1,822M. High Risk tier carries ₹3,217M in expected loss.

---

### 2️⃣ Credit Risk Driver Analysis
![Dashboard 2](POWER%20BI%20DASHBOARD/dashboard2_risk_driver_analysis.png)
> High Risk borrowers default at 12.8% vs 10.4% for Low Risk. Low-income segment shows 17.4% default rate. Unemployed applicants default at 13.6% vs 9.5% for Full-time employees.

---

### 3️⃣ ML Model Output & Risk Scoring
![Dashboard 3](POWER%20BI%20DASHBOARD/dashboard3_ml_model_output.png)
> 255K loans scored — 39% Auto Approved, 42% Manual Review, 19% Auto Rejected. Model accurately separates defaulters from non-defaulters across all probability bands.

---

### 4️⃣ Credit Policy Simulation & Pareto Analysis
![Dashboard 4](POWER%20BI%20DASHBOARD/dashboard4_pareto_simulation.png)
> High Risk tier alone accounts for 12,790 defaults vs 7,670 for Low Risk — validating Pareto concentration. Business loans carry highest expected loss across all risk tiers.

---

### 5️⃣ Business Insights & Strategic Recommendations
![Dashboard 5](POWER%20BI%20DASHBOARD/dashboard5_business_insights.png)
> 7 key business insights and 7 strategic recommendations derived from the full analysis to guide lending policy decisions.

---

## 🔬 Machine Learning Pipeline

### Data Preprocessing
- Handled missing values and outliers
- One-hot encoding for categorical variables (Education, Employment, Marital Status, Loan Purpose)
- Feature scaling using StandardScaler
- Train/Test split with stratification

### Model Development
- Trained and compared multiple classifiers: Logistic Regression, Random Forest, XGBoost, Decision Tree
- Hyperparameter tuning with GridSearchCV
- Selected best model based on ROC-AUC, Precision, Recall, and F1-Score

### 3-Tier Risk Decision Framework
| Decision | Probability Threshold | Count |
|----------|----------------------|-------|
| ✅ Auto Approve | < 0.25 | 99,050 (39%) |
| ⚠️ Manual Review | 0.25 – 0.30 | 106,700 (42%) |
| ❌ Auto Reject | > 0.30 | 49,600 (19%) |

### Explainability
- Integrated **SHAP** for feature-level explainability
- Top risk drivers identified per applicant in real-time

---

## 🗄️ SQL Analytics (10 Sections)

1. **Data Preparation & Feature Engineering** — Risk tiers, income segments, loan burden classification
2. **Portfolio Overview** — KPIs, default rates, exposure metrics
3. **Risk Distribution** — Window functions (NTILE, PERCENT_RANK)
4. **Risk Ranking** — Top defaulters (RANK, DENSE_RANK)
5. **Risk Trend Analysis** — Time-based patterns (LAG/LEAD)
6. **Cohort & Partition Analysis** — Borrower behavior over time
7. **Pareto Concentration** — 80/20 risk concentration analysis
8. **Anomaly Detection** — Outlier identification
9. **Strategy Simulation** — Business impact modeling
10. **Final Decision Framework** — Automated approval/rejection logic

---

## 🤖 AI SQL Agent

**Natural Language → SQL → Answer pipeline** built using n8n automation platform.

```
User: "What is the default rate by employment type?"
        ↓
AI Agent (Gemini LLM) generates SQL
        ↓
PostgreSQL executes query on loan_default table
        ↓
Answer: "Unemployed applicants have the highest default rate at 13.55%"
```

### Sample Questions
- *"How many applicants have defaulted?"*
- *"What is the default rate by education level?"*
- *"Give me a complete risk profile of defaulted applicants"*
- *"Which employment type has the lowest average credit score among defaulters?"*
- *"Break down default rate by loan term"*

### Setup
```bash
npx n8n                          # Start n8n locally
# Open http://localhost:5678
# Import SUB_WORKFLOW.json first, then WORKFLOW.json
# Add Gemini API key + Supabase credentials
# Activate and open chat
```

---

## 🖥️ Streamlit App

### Features
- Single applicant risk prediction with probability score
- 3-tier business decision (Approve / Review / Reject)
- Business impact calculation
- SHAP explainability — top 5 risk drivers per applicant
- Batch prediction via CSV upload

### Run Locally
```bash
git clone https://github.com/kushagra0312/Credit-Risk-Project.git
cd Credit-Risk-Project
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 Key Business Insights

- 📌 Default rate of **11.6%** significantly impacts portfolio profitability
- 📌 **Low-income borrowers (17.4%)** are the highest risk segment
- 📌 **High loan burden customers show 23% default rate** — over-leveraging signal
- 📌 **Business and Auto loans** generate negative risk-adjusted returns
- 📌 ML model flags **49.6K loans for Auto Reject** with ₹2,137M expected loss concentration
- 📌 High Risk tier validates **Pareto concentration** — 43% of borrowers, disproportionate defaults

---

## 📋 Strategic Recommendations

- Stricter credit policies for low-income and low-credit-score borrowers
- Introduce DTI thresholds to prevent over-leveraging
- Reduce exposure to Business and Auto loan segments
- Increase focus on Home loans (lowest default rate)
- Deploy 3-tier ML approval framework at 0.30 probability threshold
- Prioritize underwriting on Manual Review bucket (₹3,504M total expected loss)

---

## 👨‍💻 Author

**Kushagra Yadav** — Data Analyst
Python • SQL • Power BI • Machine Learning • AI Automation

[![GitHub](https://img.shields.io/badge/GitHub-kushagra0312-black?logo=github)](https://github.com/kushagra0312)

---

## 📄 License
MIT License
