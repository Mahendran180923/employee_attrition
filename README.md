
# Employee Attrition Prediction

This is a web application built using **Streamlit** that predicts whether an employee is likely to leave the company (attrition) based on various features such as age, job role, education, income, satisfaction levels, and more. The app uses a machine learning model trained on employee data to perform the predictions.

---

## 📁 Project Structure

- `app.py` – Streamlit-based frontend for the prediction interface.
- `attrition_prediction.py` – Contains model training, preprocessing, encoding, and optional app components.
- `model.pkl` – Trained Random Forest model.
- `encoder.pkl` – Pickle file storing the label encoders for categorical features.
- `Employee_Attrition.csv` – The dataset used for training the model (expected at `D:\Employee_Attrition`).

---

## 🚀 Features

- Load and preprocess employee data.
- Interactive UI for manually inputting employee details.
- Predict attrition likelihood using a trained Random Forest Classifier.
- User-friendly web interface using Streamlit.

---

## 🛠️ Installation

1. Clone the repository or download the project files.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Ensure the following files are located in `D:\Employee_Attrition\`:
   - `Employee_Attrition.csv`
   - `model.pkl`
   - `encoder.pkl`

> You can modify the file paths in the scripts to suit your directory structure.

---

## 💡 How to Run the App

Navigate to the project directory and run:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

---

## ⚙️ Model Information

- **Model Used**: Random Forest Classifier
- **Other Models Evaluated**: Gradient Boosting Classifier, Decision Tree Classifier
- **Evaluation Metrics**:
  - Accuracy Score
  - Mean Absolute Error
  - Root Mean Squared Error
  - R² Score

---

## 📊 Data Preprocessing

- Dropped irrelevant columns such as `EmployeeNumber`, `EmployeeCount`, `Over18`, and `StandardHours`.
- Label encoded all categorical features.
- Checked for outliers using Z-scores.
- Data split using an 80/20 train/test ratio.


---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.

---
