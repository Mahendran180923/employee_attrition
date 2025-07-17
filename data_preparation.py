import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting data preparation process (cleaning, feature engineering, encoding, EDA)...")

# --- Load Dataset ---
try:
    df = pd.read_csv("Employee_Attrition.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Employee_Attrition.csv' not found. Please ensure it's in the same directory.")
    exit()

# Store original column names before any transformations for later reference
original_columns = df.columns.tolist()

# --- Data Cleaning and Preprocessing ---
print("\nApplying data cleaning and preprocessing...")

duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Impute NumCompaniesWorked where it's 0 but total working years > company years by exactly 1
condition = ((df['NumCompaniesWorked'] == 0) & ((df['TotalWorkingYears'] - df['YearsAtCompany']) == 1))
df.loc[condition, 'NumCompaniesWorked'] = 1

df['ManagerTenureRatio'] = df['YearsWithCurrManager'] / df['YearsAtCompany'].replace(0, np.nan)
df['ManagerTenureRatio'] = df['ManagerTenureRatio'].fillna(0)

# --- Feature Engineering ---
print("\nApplying feature engineering...")

df['TenureCategory'] = pd.cut(df['YearsAtCompany'],
                              bins=[-1, 2, 5, 10, np.inf],
                              labels=['<2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs'])

df['PerformanceMetric'] = df['PerformanceRating'] * df['PercentSalaryHike']

# EngagementScore (JobSatisfaction removed from calculation as per previous fix)
df['EngagementScore'] = (
    df['EnvironmentSatisfaction'] +
    df['RelationshipSatisfaction'] +
    df['JobInvolvement']
)

df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']

df['YearsPerCompany'] = df['TotalWorkingYears'] / df['NumCompaniesWorked']
df['YearsPerCompany'] = df['YearsPerCompany'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['YearsSinceChange'] = df['YearsAtCompany'] - df['YearsInCurrentRole']

df['YearsInSameRole'] = np.where(
    df['YearsAtCompany'] == 0,
    0,
    df['YearsInCurrentRole'] / df['YearsAtCompany']
)

df['TrainingPerYear'] = np.where(
    df['YearsAtCompany'] == 0,
    0,
    df['TrainingTimesLastYear'] / df['YearsAtCompany']
)

df['YearsInSameRole'] = df['YearsInSameRole'].round(3)
df['TrainingPerYear'] = df['TrainingPerYear'].round(3)

df.loc[
    (df['NumCompaniesWorked'] == 1) &
    (df['TotalWorkingYears'] == 0) &
    (df['YearsAtCompany'] == 0),
    'NumCompaniesWorked'
] = 0


# --- Encoding ---
print("\nApplying encoding...")

all_fitted_encoders = {} # Dictionary to store all fitted encoders

# 1. Label Encoding for Binary Nominal
le_attrition = LabelEncoder()
df['Attrition'] = le_attrition.fit_transform(df['Attrition'])
# We don't save le_attrition as it's for the target variable, not features for prediction

le_overtime = LabelEncoder()
df['OverTime'] = le_overtime.fit_transform(df['OverTime'])
all_fitted_encoders['OverTime'] = le_overtime

le_over18 = LabelEncoder()
df['Over18'] = le_over18.fit_transform(df['Over18'])
all_fitted_encoders['Over18'] = le_over18


# 2. Ordinal Encoding for BusinessTravel and TenureCategory
oe_businesstravel = OrdinalEncoder(categories=[['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']])
df['BusinessTravel'] = oe_businesstravel.fit_transform(df[['BusinessTravel']])
all_fitted_encoders['BusinessTravel'] = oe_businesstravel

tenure_order = ['<2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs']
oe_tenure = OrdinalEncoder(categories=[tenure_order])
df['TenureCategory'] = oe_tenure.fit_transform(df[['TenureCategory']])
all_fitted_encoders['TenureCategory'] = oe_tenure


# 3. One Hot Encoding
# These are the columns that will be one-hot encoded.
onehot_cols = ['EducationField', 'Gender', 'Department', 'JobRole', 'MaritalStatus']
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Create the ColumnTransformer
# This CT will be used to transform new data consistently
preprocessor_ct = ColumnTransformer(
    transformers=[
        ('onehot', ohe, onehot_cols)
    ],
    remainder='passthrough' # Keep all other columns (numerical, already encoded binary/ordinal)
)

# Fit and transform the DataFrame using the ColumnTransformer
transformed_array = preprocessor_ct.fit_transform(df)

# *** CRITICAL FIX: Get all output feature names directly from the ColumnTransformer ***
# This ensures the column names exactly match the transformed array's shape and order.
final_feature_names_all_processed_df = preprocessor_ct.get_feature_names_out()

df_processed = pd.DataFrame(transformed_array, columns=final_feature_names_all_processed_df)

# The target variables 'Attrition' and 'JobSatisfaction' are now part of df_processed
# because they were passed through by the ColumnTransformer (as they were not in onehot_cols).
# So, no need to re-add them from the original 'df'.
# We just need to ensure their data types are correct if they were numeric originally.


all_fitted_encoders['ColumnTransformer'] = preprocessor_ct
all_fitted_encoders['final_feature_names_all_processed_df'] = final_feature_names_all_processed_df # Store for later use

print("Processed DataFrame head (all features, post-encoding):")
print(df_processed.head())
print("Processed DataFrame shape (all features, post-encoding):", df_processed.shape)
print("Processed DataFrame columns (all features, post-encoding):", df_processed.columns.tolist())

# --- EDA: Outlier Detection ---
print("\nPerforming Outlier Detection (EDA)...")
numerical_columns = df_processed.select_dtypes(include='number').columns.to_list()
# Exclude target variables from outlier detection if they are not features
numerical_columns_for_outliers = [col for col in numerical_columns if col not in ['remainder__Attrition', 'remainder__JobSatisfaction']]

outlier_summary = {}
for col in numerical_columns_for_outliers:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
    outlier_summary[col] = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'num_outliers': outliers.shape[0]
    }

outliers_df = pd.DataFrame(outlier_summary).T.sort_values(by='num_outliers', ascending=False)
outliers_df.reset_index(inplace=True)
outliers_df.rename(columns={'index': 'column_name'}, inplace=True)
print("Outlier Summary (IQR method):")
print(outliers_df)

# Z-score method for outliers
print("\nOutlier Detection (Z-score method):")
for col in numerical_columns_for_outliers:
    std = df_processed[col].std()
    if std == 0:
        continue
    z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / std)
    outliers = df_processed[z_scores > 4] # Threshold of 4 for Z-score
    print(f"Outliers in column '{col}' (Z-score > 4): {outliers.shape[0]}")


# --- EDA: Correlation Heatmap ---
print("\nGenerating Correlation Heatmap (EDA)...")
# Select only numeric columns for correlation matrix
numeric_cols_for_corr = df_processed.select_dtypes(include=np.number).columns.tolist()
# Exclude target variables if you don't want them in the feature correlation heatmap
numeric_cols_for_corr_features = [col for col in numeric_cols_for_corr if col not in ['remainder__Attrition', 'remainder__JobSatisfaction']]

corr_matrix = df_processed[numeric_cols_for_corr_features].corr()
plt.figure(figsize=(22, 18), dpi=120)
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    square=False,
    linewidths=.5,
    cbar_kws={'shrink': .6},
    annot=False # Set to True for values, but can be cluttered for many features
)
plt.xticks(rotation=90, ha='center', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.title('Correlation Heatmap (Numeric Features after Preprocessing)')
plt.tight_layout()
plt.show()


# --- Save Processed Data and All Fitted Encoders ---
print("\nSaving processed data and all fitted encoders...")
try:
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(df_processed, f)
    print("Processed DataFrame saved as 'processed_data.pkl'.")

    with open('all_fitted_encoders.pkl', 'wb') as f:
        pickle.dump(all_fitted_encoders, f)
    print("All fitted encoders and ColumnTransformer saved as 'all_fitted_encoders.pkl'.")
except Exception as e:
    print(f"Error saving files: {e}")

print("\nData preparation process completed successfully!")