import mlflow
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load dataset
employee_data = pd.read_csv('D:\Projects\Guvi\Employee_Attrition\Employee_Attrition.csv')


# Data Cleaning and Preprocessing

df = pd.DataFrame(employee_data)
# print(df.info())
# print(df.describe())

duplicates = df.duplicated().sum()
# print(f"Number of duplicate rows: {duplicates}")
df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)


# Exploratory data analysis

# Encode the object columns
encoder = {}
categorical_col = df.select_dtypes(include='object').columns
for col in categorical_col:
    encoder[col] = LabelEncoder()
    df[col] = encoder[col].fit_transform(df[col])





# Outlier detecton
for col in df.columns:
    z_scores = np.abs(df[col] - df[col].mean() / df[col].std())
    outliers = df[z_scores > 4]
    # print(f"Outliers in column '{col}': {outliers.shape[0]}")
    # print(outliers)


# Find correlation 
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, cmap='coolwarm', square=True)
# plt.title('Correlation Heatmap')
# plt.show()


# Machine Learning Model development

# Feature Selection
x = df.drop(['Attrition'], axis=1)
y = df['Attrition']


# Various Model Training
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.20, random_state=42)

with mlflow.start_run():
    # Random Forest Classifier Model
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(x_train, y_train)
    # prediction = rfc.predict(x_test)

    # # Gradient Boosting Classifier Model
    gbc = GradientBoostingClassifier(learning_rate=0.1)
    gbc.fit(x_train, y_train)
    gbc_prediction = gbc.predict(x_test)

    # # Decision Tree Classifier Model
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    dtc_prediction = dtc.predict(x_test)

    # Model Evaluation Metrics
    rfc_score = rfc.score(x_test, y_test)
    gbc_score = gbc.score(x_test, y_test)
    dtc_score = dtc.score(x_test, y_test)
    # rfc_r2_score = r2_score(x_test,  prediction)

    # gbc_mae_score = mean_absolute_error(x_test,  gbc_prediction)
    # gbc_mse_score = mean_squared_error(x_test,  gbc_prediction)
    # gbc_rmse_score = root_mean_squared_error(x_test,  gbc_prediction)
    # gbc_r2_score = r2_score(x_test,  gbc_prediction)

    # dtc_mae_score = mean_absolute_error(x_test,  dtc_prediction)
    # dtc_mse_score = mean_squared_error(x_test,  dtc_prediction)
    # dtc_rmse_score = root_mean_squared_error(x_test,  dtc_prediction)
    # dtc_r2_score = r2_score(x_test,  dtc_prediction)


    #log the metrics and parameter into mlflow
    mlflow.log_metric("rfc_mae_score", rfc_score)
    # mlflow.log_metric("rfc_mse_score", rfc_mse_score)
    # mlflow.log_metric("rfc_rmse_score", rfc_rmse_score)
    # mlflow.log_metric("rfc_r2_score", rfc_r2_score)

    mlflow.log_metric("gbc_mae_score", gbc_score)
    # mlflow.log_metric("gbc_mse_score", gbc_mse_score)
    # mlflow.log_metric("gbc_rmse_score", gbc_rmse_score)
    # mlflow.log_metric("gbc_r2_score", gbc_r2_score)

    mlflow.log_metric("dtc_mae_score", dtc_score)
    # mlflow.log_metric("dtc_mse_score", dtc_mse_score)
    # mlflow.log_metric("dtc_rmse_score", dtc_rmse_score)
    # mlflow.log_metric("dtc_r2_score", dtc_r2_score)

    #log the model into ml
    mlflow.sklearn.log_model(rfc,"rfc_model")
    mlflow.sklearn.log_model(gbc,"gbc_model")
    mlflow.sklearn.log_model(dtc,"dtc_model")


print("MLflow run completed.")



