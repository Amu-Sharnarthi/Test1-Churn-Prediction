🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

📌 Project Overview
This project presents an interactive Streamlit dashboard for predicting and analyzing customer churn in a European bank. It combines machine learning models, explainability techniques, and business insights to help banks take proactive retention actions.
The dashboard allows users to:
•	Predict churn probability for individual customers 
•	Understand key drivers behind churn 
•	Analyze model performance 
•	Simulate “what-if” scenarios 
•	Plan targeted retention strategies 
________________________________________
🎯 Objectives
Primary Objectives
•	Predict customer churn with high accuracy 
•	Generate churn probability scores 
•	Identify key drivers influencing churn 
Secondary Objectives
•	Improve model interpretability using SHAP 
•	Reduce false positives 
•	Enable actionable business decisions 
________________________________________
⚙️ Tech Stack
•	Frontend: Streamlit 
•	Backend / ML: Scikit-learn 
•	Visualization: Plotly, Matplotlib, Seaborn 
•	Explainability: SHAP 
•	Data Handling: Pandas, NumPy 
•	Model Storage: Joblib 
________________________________________
🧠 Machine Learning Models
•	🌲 Random Forest Classifier 
•	⚡ Gradient Boosting Classifier 
Both models are built using pipelines that include:
•	Preprocessing (encoding, scaling) 
•	Feature engineering 
•	Classification 
________________________________________
📊 Key Features of Dashboard
1. 🔍 Overview
•	Churn Probability 
•	Risk Score (0–100) 
•	Risk Category (Low / Medium / High) 
•	Retention Rate 
•	Gauge visualization 
________________________________________
2. 📈 Segmentation
•	Customer segmentation (Low / Medium / High risk) 
•	Feature importance visualization 
________________________________________
3. 📉 Model Performance
•	Accuracy 
•	Precision 
•	Recall 
•	F1 Score 
•	ROC AUC 
________________________________________
4. 🔬 Explainability
•	SHAP-based local explanations 
•	Feature impact on churn prediction 
•	Geography-wise churn analysis 
________________________________________
5. 📌 Partial Dependence (PDP)
•	Shows how features affect churn probability: 
o	Age 
o	Balance 
o	Number of Products 
o	Active Membership 
o	Credit Score 
________________________________________
6. 🔄 What-if Simulator
•	Modify: 
o	Credit Score 
o	Balance 
o	Number of Products 
•	Instantly see impact on churn probability 
________________________________________
7. 🎯 Retention Strategy Module
•	Suggests best actions: 
o	Call Relationship Manager 
o	Send Personalized Email 
o	Offer Fee Waiver 
o	Offer Loyalty Bonus 
________________________________________
🧩 Feature Engineering
The model uses additional engineered features:
•	Balance_Salary_Ratio 
•	Product_Density 
•	Engagement_Product 
•	Age_Tenure_Interaction 
These enhance prediction accuracy and business insight.
________________________________________
📂 Project Structure
├── EUChurn_Streamlit.py
├── rf_churn_prediction_model.pkl
├── gb_churn_prediction_model.pkl
├── European_Bank_Data.csv
├── EULOGO.png
├── UMlogo.png
└── README.md
________________________________________
🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run Streamlit App
streamlit run EUChurn_Streamlit.py
________________________________________
📥 Input Parameters
The dashboard accepts:
•	Geography (France, Spain, Germany) 
•	Gender 
•	Credit Score 
•	Age 
•	Tenure 
•	Balance 
•	Number of Products 
•	Credit Card Status 
•	Active Membership 
•	Estimated Salary 
________________________________________
📤 Outputs
•	Churn Probability 
•	Risk Score 
•	Risk Category 
•	Retention Recommendations 
•	Visual Insights 
________________________________________
🛡️ Error Handling
•	Safe model loading using custom patch for sklearn compatibility 
•	Graceful fallback for missing files 
•	Input parsing for numeric values 
________________________________________
📌 Use Cases
•	Banking customer retention 
•	CRM optimization 
•	Risk segmentation 
•	Marketing targeting 
•	Business intelligence dashboards 
________________________________________
👨‍🎓 Author
Ambika Sharnarthi
________________________________________
👨‍🏫 Guided By
Sai Kagne
________________________________________
📜 License
This project is for academic and educational purposes.
________________________________________

