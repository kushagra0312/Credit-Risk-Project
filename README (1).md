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
│
├── 🤖 best_credit_risk_model.pkl            # Trained ML model (Random Forest / XGBoost)
├── ⚖️  scaler.pkl                            # Feature scaler for preprocessing
│
├── 🖥️  app.py                               # Streamlit web application
│
├── 🗄️  CREDIT_RISK_FINAL.sql               # Advanced SQL analytics (10 sections)
│
├── 📊 Power BI Dashboards/
│   ├── Loan Portfolio Performance Overview
│   ├── Credit Risk Driver Analysis
│   ├── ML Model Output & Risk Scoring
│   └── Business Insights & Strategic Recommendations
│
├── 🤖 AI Agent/
│   ├── CREDIT_RISK_SQL_AI_AGENT_WORKFLOW.json      # Main n8n workflow
│   └── CREDIT_RISK_SQL_AI_AGENT_SUB_WORKFLOW.json  # Sub-workflow (Postgres executor)
│
├── requirements.txt
└── README.md
```

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

Advanced SQL analysis covering the full credit risk lifecycle:

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

## 📈 Power BI Dashboards

### Dashboard 1: Loan Portfolio Performance Overview
- Total disbursed, default count, risk exposure KPIs
- Default rate by loan purpose and risk tier
- Expected loss heatmap by loan purpose
- Expected loss by model decision (Auto Approve / Reject / Manual Review)

### Dashboard 2: Credit Risk Driver Analysis
- Default rate by credit risk category (High/Medium/Low)
- Default rate by income segment and employment type
- Loan burden impact on defaults
- Employment stability vs default rate

### Dashboard 3: ML Model Output & Risk Scoring
- Model decision funnel (donut chart)
- Predicted probability distribution
- Predicted probability vs actual default rate scatter
- Expected loss by model decision and risk tier

### Dashboard 4: Business Insights & Strategic Recommendations
- 7 key business insights derived from data
- 7 actionable strategic recommendations for lending policy

---

## 🤖 AI SQL Agent

**Natural Language → SQL → Answer pipeline** built using n8n automation platform.

### How It Works
```
User: "What is the default rate by employment type?"
        ↓
AI Agent (Gemini LLM) generates SQL
        ↓
PostgreSQL executes query on loan_default table
        ↓
Answer: "Unemployed applicants have the highest default rate at 13.55%"
```

### Sample Questions the Agent Can Answer
- *"How many applicants have defaulted?"*
- *"What is the default rate by education level?"*
- *"Give me a complete risk profile of defaulted applicants"*
- *"Which employment type has the lowest average credit score among defaulters?"*
- *"Break down default rate by loan term"*

### Setup Instructions
1. Install n8n: `npx n8n`
2. Open `http://localhost:5678`
3. Import `CREDIT_RISK_SQL_AI_AGENT_SUB_WORKFLOW.json` first
4. Import `CREDIT_RISK_SQL_AI_AGENT_WORKFLOW.json`
5. Add your Gemini API key and Supabase/PostgreSQL credentials
6. Activate workflow and open chat

---

## 🖥️ Streamlit App — Live Demo

### Features
- Single applicant risk prediction with probability score
- 3-tier business decision (Approve / Review / Reject)
- Business impact calculation (Loss avoided / Extra revenue)
- SHAP-based explainability showing top 5 risk drivers
- Batch prediction via CSV upload with downloadable results

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
- 📌 **Low-income borrowers (17.4%)** and **high-risk credit score borrowers (12.8%)** are primary default contributors
- 📌 Customers with **high loan burden show 23% default rate** — indicating over-leveraging
- 📌 **Business and Auto loans** generate negative risk-adjusted returns
- 📌 ML model flags **49.6K loans for Auto Reject**, concentrating ₹2,137M in expected loss
- 📌 High Risk tier accounts for disproportionately higher defaults — validating **Pareto concentration of credit risk**

---

## 📋 Strategic Recommendations

- Implement stricter credit policies for low-income and low-credit-score borrowers
- Introduce loan-to-income and DTI thresholds to prevent over-leveraging
- Reduce exposure to high-risk segments (Business and Auto loans)
- Increase focus on low-risk profitable segments (Home loans)
- Incorporate employment stability checks into loan approval processes
- Deploy 3-tier ML approval framework based on predicted default probability threshold of 0.30
- Prioritize underwriting resources on Manual Review bucket (106.7K loans carrying ₹3,504M total expected loss)

---

## 👨‍💻 Author

**Kushagra Yadav**
Data Analyst | Python • SQL • Power BI • Machine Learning • AI Automation

---

## 📄 License

This project is licensed under the MIT License.
