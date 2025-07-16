from attrition_prediction import employee_data
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

file_path = os.path.join("D:", "Employee_Attrition", "rfc_attrition_model.pkl")



option = st.selectbox('Select employee number to fetch employee details or select details manually', ['Select Manually', 'Use Employee Number'])

if option == 'Select Manually':
    Age = st.number_input('Age', 18, 60)
    BusinessTravel = st.selectbox('BusinessTravel', employee_data['BusinessTravel'].unique())
    DailyRate = st.select_slider('DailyRate', sorted(employee_data['DailyRate'].unique()))
    Department = st.selectbox('Department', employee_data['Department'].unique())
    DistanceFromHome = st.select_slider('DistanceFromHome', sorted(employee_data['DistanceFromHome'].unique()))
    Education = st.selectbox('Education: Select "1" for below college or select "2" for above college ', employee_data['Education'].unique())
    EducationField = st.selectbox('EducationField', employee_data['EducationField'].unique())
    EnvironmentSatisfaction = st.selectbox('Satisfaction with the work environment: "1" = Low, "2" = Medium, "3" = High, "4" = Very High', sorted(employee_data['EnvironmentSatisfaction'].unique()))
    Gender = st.selectbox('Gender', employee_data['Gender'].unique())
    HourlyRate = st.slider('The employee hourly rate of pay', min(employee_data['HourlyRate'].unique()), max(employee_data['HourlyRate'].unique()))
    JobInvolvement = st.selectbox('level of involvement the employee has in their job: "1" = Low, "2" = Medium, "3" = High, "4" = Very High', sorted(employee_data['JobInvolvement'].unique()) )
    JobLevel = st.selectbox('Job level of the Employee : e.g., "1" = Entry Level, "2" = Mid-Level, etc', sorted(employee_data['JobLevel'].unique()) )
    JobRole = st.selectbox('JobRole', employee_data['JobRole'].unique())
    JobSatisfaction = st.selectbox('Job Satisfaction with the job: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['JobSatisfaction'].unique()) )
    MaritalStatus = st.selectbox('MaritalStatus', employee_data['MaritalStatus'].unique())
    MonthlyIncome = st.slider('MonthlyIncome', min(employee_data['MonthlyIncome'].unique()), max(employee_data['MonthlyIncome'].unique()))
    MonthlyRate = st.slider('MonthlyRate', min(employee_data['MonthlyRate'].unique()), max(employee_data['MonthlyRate'].unique()))
    NumCompaniesWorked = st.select_slider('NumCompaniesWorked', sorted(employee_data['NumCompaniesWorked'].unique()))
    OverTime = st.selectbox('OverTime', employee_data['OverTime'].unique())
    PercentSalaryHike = st.selectbox('PercentSalaryHike', sorted(employee_data['PercentSalaryHike'].unique()))
    PerformanceRating = st.selectbox('Peformance Rating: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['PerformanceRating'].unique()))
    RelationshipSatisfaction = st.selectbox('Relationship Satisfaction: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['RelationshipSatisfaction'].unique()))
    StockOptionLevel = st.number_input('Stock Option Level', min(employee_data['StockOptionLevel'].unique()), max(employee_data['StockOptionLevel'].unique()))
    TotalWorkingYears = st.selectbox('TotalWorkingYears', sorted(employee_data['TotalWorkingYears'].unique()))
    TrainingTimesLastYear = st.selectbox('TrainingTimesLastYear', sorted(employee_data['TrainingTimesLastYear'].unique()))
    YearsAtCompany = st.selectbox('YearsAtCompany', sorted(employee_data['YearsAtCompany'].unique()))
    WorkLifeBalance = st.selectbox('Work Life Balance: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['WorkLifeBalance'].unique()))
    YearsInCurrentRole = st.selectbox('YearsInCurrentRole', sorted(employee_data['YearsInCurrentRole'].unique()))
    YearsSinceLastPromotion = st.selectbox('YearsSinceLastPromotion', sorted(employee_data['YearsSinceLastPromotion'].unique()))
    YearsWithCurrManager = st.selectbox('YearsWithCurrManager', sorted(employee_data['YearsWithCurrManager'].unique()))


user_data = pd.DataFrame(
    {'Age': [Age], 
    'BusinessTravel': [BusinessTravel], 
    'DailyRate': [DailyRate],  
    'Department': [Department],  
    'DistanceFromHome': [DistanceFromHome],  
    'Education': [Education],  
    'EducationField': [EducationField],  
    'EnvironmentSatisfaction': [EnvironmentSatisfaction],  
    'Gender': [Gender],  
    'HourlyRate': [HourlyRate],  
    'JobInvolvement': [JobInvolvement],  
    'JobLevel': [JobLevel],  
    'JobRole': [JobRole],  
    'JobSatisfaction': [JobSatisfaction],  
    'MaritalStatus': [MaritalStatus],  
    'MonthlyIncome': [MonthlyIncome],  
    'MonthlyRate': [MonthlyRate],  
    'NumCompaniesWorked': [NumCompaniesWorked],  
    'OverTime': [OverTime],  
    'PercentSalaryHike': [PercentSalaryHike],  
    'PerformanceRating': [PerformanceRating],  
    'RelationshipSatisfaction': [RelationshipSatisfaction],  
    'StockOptionLevel': [StockOptionLevel],  
    'TotalWorkingYears': [TotalWorkingYears],  
    'TrainingTimesLastYear': [TrainingTimesLastYear],  
    'YearsAtCompany': [YearsAtCompany],  
    'WorkLifeBalance ': [WorkLifeBalance], 
    'YearsInCurrentRole': [YearsInCurrentRole],  
    'YearsSinceLastPromotion': [YearsSinceLastPromotion],  
    'YearsWithCurrManager': [YearsWithCurrManager],})

with open(file_path, 'rb') as f:
    reloaded_rfc_attrition_prediction = pickle.load(f)

with open("D:\Employee_Attrition\attrition_encoder.pkl", 'rb') as f:
    reloaded_attrition_encoder = pickle.load(f)

for col in user_data.columns:
    if col in reloaded_attrition_encoder:
        user_data[col] = reloaded_attrition_encoder[col].transform(user_data[col])

if st.button('Predict Employee Attrition'):
    prediction = reloaded_rfc_attrition_prediction.predict(user_data)

    st.write(f"Employee Prediction{prediction}")

