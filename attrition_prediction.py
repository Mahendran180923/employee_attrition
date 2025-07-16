import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score, root_mean_squared_error
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load dataset
employee_data = pd.read_csv('D:\Employee_Attrition\Employee_Attrition.csv')


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


# Random Forest Classifier Model
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(x_train, y_train)

# Gradient Boosting Classifier Model
gbc = GradientBoostingClassifier(learning_rate=0.1)
gbc.fit(x_train, y_train)

# Decision Tree Classifier Model
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

# Model Evaluation Metrics
rfc_score = rfc.score(x_test, y_test)
gbc_score = gbc.score(x_test, y_test)
dtc_score = dtc.score(x_test, y_test)

print(f'Random Forest Accuracy Score: {rfc_score}')
print(f'Gradient Boosing Accuracy Score: {gbc_score}')
print(f'Decision Tree Accuracy Score: {dtc_score}')

# Store the encoded file in pickle format
with open("encoder.pkl", 'wb') as f:
    pickle.dump(encoder, f)


# Save the model as pickle file
with open("model.pkl", 'wb') as f:
  pickle.dump(rfc, f)

# option = st.selectbox('Select employee number to fetch employee details or select details manually', ['Select Manually', 'Use Employee Number'])

# # if option == 'Select Manually':
Age = st.number_input('Age', 18, 60)
BusinessTravel = st.selectbox('BusinessTravel', employee_data['BusinessTravel'].unique())
DailyRate = st.selectbox('DailyRate', sorted(employee_data['DailyRate'].unique()))
Department = st.selectbox('Department', employee_data['Department'].unique())
DistanceFromHome = st.selectbox('DistanceFromHome', sorted(employee_data['DistanceFromHome'].unique()))
Education = st.selectbox('Education: Select "1" for below college or select "2" for above college ', employee_data['Education'].unique())
EducationField = st.selectbox('EducationField', employee_data['EducationField'].unique())
EnvironmentSatisfaction = st.selectbox('Satisfaction with the work environment: "1" = Low, "2" = Medium, "3" = High, "4" = Very High', sorted(employee_data['EnvironmentSatisfaction'].unique()))
Gender = st.selectbox('Gender', employee_data['Gender'].unique())
HourlyRate = st.selectbox('The employee hourly rate of pay', sorted(employee_data['HourlyRate'].unique()))
JobInvolvement = st.selectbox('level of involvement the employee has in their job: "1" = Low, "2" = Medium, "3" = High, "4" = Very High', sorted(employee_data['JobInvolvement'].unique()) )
JobLevel = st.selectbox('Job level of the Employee : e.g., "1" = Entry Level, "2" = Mid-Level, etc', sorted(employee_data['JobLevel'].unique()) )
JobRole = st.selectbox('JobRole', employee_data['JobRole'].unique())
JobSatisfaction = st.selectbox('Job Satisfaction with the job: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['JobSatisfaction'].unique()) )
MaritalStatus = st.selectbox('MaritalStatus', employee_data['MaritalStatus'].unique())
MonthlyIncome = st.selectbox('MonthlyIncome', sorted(employee_data['MonthlyIncome'].unique()))
MonthlyRate = st.selectbox('MonthlyRate', sorted(employee_data['MonthlyRate'].unique()))
NumCompaniesWorked = st.selectbox('NumCompaniesWorked', sorted(employee_data['NumCompaniesWorked'].unique()))
OverTime = st.selectbox('OverTime', employee_data['OverTime'].unique())
PercentSalaryHike = st.selectbox('PercentSalaryHike', sorted(employee_data['PercentSalaryHike'].unique()))
PerformanceRating = st.selectbox('Peformance Rating: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['PerformanceRating'].unique()))
RelationshipSatisfaction = st.selectbox('Relationship Satisfaction: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['RelationshipSatisfaction'].unique()))
StockOptionLevel = st.number_input('Stock Option Level', min(employee_data['StockOptionLevel'].unique()), max(employee_data['StockOptionLevel'].unique()))
TotalWorkingYears = st.selectbox('TotalWorkingYears', sorted(employee_data['TotalWorkingYears'].unique()))
TrainingTimesLastYear = st.selectbox('TrainingTimesLastYear', sorted(employee_data['TrainingTimesLastYear'].unique()))
WorkLifeBalance = st.selectbox('Work Life Balance: "1"=Low, "2"=Medium, "3"=High, "4"=Very High', sorted(employee_data['WorkLifeBalance'].unique()))
YearsAtCompany = st.selectbox('YearsAtCompany', sorted(employee_data['YearsAtCompany'].unique()))
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
    'WorkLifeBalance': [WorkLifeBalance],
    'YearsAtCompany': [YearsAtCompany],  
    'YearsInCurrentRole': [YearsInCurrentRole],  
    'YearsSinceLastPromotion': [YearsSinceLastPromotion],  
    'YearsWithCurrManager': [YearsWithCurrManager],})

with open("D:\Employee_Attrition\model.pkl", 'rb') as f:
    reloaded_rfc_attrition_prediction = pickle.load(f)

with open("D:\Employee_Attrition\encoder.pkl", 'rb') as f:
    reloaded_attrition_encoder = pickle.load(f)

for col in user_data.columns:
    if col in reloaded_attrition_encoder:
        user_data[col] = reloaded_attrition_encoder[col].transform(user_data[col])

if st.button('Predict Employee Attrition'):
    prediction = reloaded_rfc_attrition_prediction.predict(user_data)
    predicted_label = reloaded_attrition_encoder['Attrition'].inverse_transform(prediction)
    st.write(f"Employee Prediction: {predicted_label[0]}")
    
