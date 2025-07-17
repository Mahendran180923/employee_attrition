
# 📊 Employee Attrition & Job Satisfaction Prediction Dashboard

A complete end-to-end machine learning project that predicts **employee attrition** and **job satisfaction**, powered by interactive visualizations and model explainability in **Streamlit**.

---

## 🚀 Features

- 🔍 Attrition prediction using Logistic Regression.
- 📈 Job satisfaction score prediction using Random Forest Regressor.
- 🧠 Feature Engineering and Label/OneHot/Ordinal Encoding.
- 📊 KPI Dashboard with satisfaction and attrition analysis.
- 📁 Modular Python script structure.
- 🎨 Interactive dashboard built using **Streamlit**.
- 📦 Models and encoders are saved using Pickle for fast deployment.

---

## 🗂️ Project Structure

```
├── attrition_model_training.py               # Train attrition prediction models
├── satisfaction_model_training.py            # Train job satisfaction regression models
├── data_preparation.py                       # Cleans, encodes and preprocesses the dataset
├── data_visualization_analysis.py            # Generates KPI and visual plots
├── streamlit_app.py                          # Streamlit web application
├── Employee_Attrition.csv                    # Input dataset
├── all_fitted_encoders.pkl                   # Saved LabelEncoder, OrdinalEncoder, ColumnTransformer
├── attrition_logistic_regression.pkl         # Attrition model
├── satisfaction_random_forest_regressor.pkl  # Satisfaction model
├── attrition_model_features.pkl              # Attrition model's feature set
├── satisfaction_model_features.pkl           # Satisfaction model's feature set
├── processed_data.pkl                        # Processed dataset
├── plots/
│   ├── kpis.pkl
│   ├── attrition_comparison_df.pkl
│   └── satisfaction_comparison_df.pkl
└── README.md                                 # Project documentation
```

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/employee-analytics-dashboard.git
cd employee-analytics-dashboard
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate    # On Windows
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, use:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 3. Prepare the Data

```bash
python data_preparation.py
```

### 4. Train the Models

```bash
python attrition_model_training.py
python satisfaction_model_training.py
```

### 5. Generate KPIs and Charts

```bash
python data_visualization_analysis.py
```

### 6. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📌 Notes

- All models, encoders, and visualizations are saved in `.pkl` format.
- Make sure to run all training and preprocessing scripts before launching the app.
- You can use `git add . && git commit -m "your message" && git push` to update changes to GitHub.

---

## 👤 Author

**Mahendran S**  
MIS & Costing Executive | Aspiring Data Scientist  
📧 mahendran.s15593@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mahendran-sudalai-00182b294)
