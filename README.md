# Customer Churn Prediction - Machine Learning Case Study

Predicting telecom customer churn using Logistic Regression and Random Forest.  
This project combines ML modeling, business insight and stakeholder recommendations to reduce churn and increase retention.

---

## 📊 Business Objective

- **Problem:** High churn impacts revenue and growth.  
- **Goal:** Predict which customers are likely to churn and understand why.  
- **Stakeholder:** Marketing Manager / Customer Retention Lead

---

📩 **From: Dave, Chief Churn Destroyer**

> "Team, I’m tired of watching customers vanish faster than snacks at a strategy meeting.  
> If we don’t fix churn, I’ll have to sell my espresso machine to make revenue look good.  
> Someone find the root cause or I’m renaming our company ‘GoodbyeTelco’!"

This project responds to Dave’s dramatic (but totally reasonable) concerns using machine learning and business analytics.

---

## 🛠 Tools Used

- Python (Pandas, scikit-learn, Matplotlib, Seaborn)
- Machine Learning: Logistic Regression, Random Forest
- Data Source: [Telco Customer Churn (Kaggle)]([https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/discussion?sort=hotness))

---

## 🔍 Key Features Driving Churn

| Feature                          | Business Insight                        |
|----------------------------------|------------------------------------------|
| `TotalCharges`, `MonthlyCharges`| High-spenders are at risk                |
| `tenure`                         | New customers churn more                 |
| `InternetService_Fiber optic`   | Fiber users churn more than DSL                   |
| `PaymentMethod_Electronic check`| Higher churn with this method            |
| `OnlineSecurity`, `TechSupport` | Reduce churn when enabled                |

---

## 📈 Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Logistic Regression | 80.5%    | 0.65      | 0.57   | 0.61     | 0.84 |
| Random Forest       | 78.9%    | 0.63      | 0.52   | 0.57     | 0.82 |

---

## 💡 Business Recommendations

- Target **new users (< 6 months)** with onboarding offers  
- Provide **bundled discounts** for high MonthlyCharges  
- Promote **auto-pay or credit card options** over electronic checks  
- Upsell **Online Security** and **Tech Support**  
- Review **Fiber optic service experience**

---

## 💼 Business Value

> Reducing churn by even 20% could save **£1M+ in customer lifetime value**.  
> ML-powered targeting improves marketing ROI and retention efforts.

---

## 📎 Deliverables

- Notebook: `Churn_Prediction_Model.py`  
- Visuals: Confusion Matrix, ROC Curves, Feature Importance  
- PDF: 1-page project summary (see `Customer Churn Prediction - Machine Learning Case Study.docx`)
