import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # For regression
from sklearn.tree import DecisionTreeRegressor # For regression
from sklearn.linear_model import LinearRegression # For regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting Job Satisfaction Model Training and Evaluation...")

# --- Load Processed Data and Encoders ---
try:
    with open('processed_data.pkl', 'rb') as f:
        df_processed = pickle.load(f)
    print("Processed DataFrame loaded successfully.")

    with open('all_fitted_encoders.pkl', 'rb') as f:
        all_fitted_encoders = pickle.load(f)
    print("All fitted encoders loaded successfully.")

    # Extract necessary info from all_fitted_encoders
    preprocessor_ct = all_fitted_encoders['ColumnTransformer']
    final_feature_names_all_processed_df = all_fitted_encoders['final_feature_names_all_processed_df']

except FileNotFoundError:
    print("Error: 'processed_data.pkl' or 'all_fitted_encoders.pkl' not found.")
    print("Please run 'data_preparation.py' first to generate these files.")
    exit()
except Exception as e:
    print(f"Error loading processed data or encoders: {e}")
    exit()


# --- Define Features (X) and Target (y) for Job Satisfaction Prediction ---
# Refactored: Using a broader set of features for Job Satisfaction prediction
# This list includes most of the available features after preprocessing,
# excluding the target variables themselves.
important_cols_user_request_satisfaction = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    # Original columns that were label/ordinal encoded
    'OverTime', 'BusinessTravel', 'TenureCategory', 'Over18',
    # Engineered features
    'PerformanceMetric', 'EngagementScore', 'PromotionGap',
    'YearsPerCompany', 'YearsSinceChange', 'YearsInSameRole', 'TrainingPerYear',
    # One-hot encoded original columns (their base names)
    'EducationField', 'Gender', 'Department', 'JobRole', 'MaritalStatus'
]


# Dynamically build the list of actual features that will be in X for satisfaction model.
# This ensures only the requested features (and their OHE forms) are included.
X_satisfaction_columns = []
# Get the original column names that were one-hot encoded by the CT
# This is used to correctly identify the OHE columns in df_processed
onehot_cols_from_ct = preprocessor_ct.named_transformers_['onehot'].feature_names_in_

for col_name_in_processed_df in final_feature_names_all_processed_df:
    # Check if this processed column is a one-hot encoded feature AND its original form is in user's request
    is_onehot_feature = False
    for original_cat_col in onehot_cols_from_ct:
        if col_name_in_processed_df.startswith(f'onehot__{original_cat_col}_') and original_cat_col in important_cols_user_request_satisfaction:
            X_satisfaction_columns.append(col_name_in_processed_df)
            is_onehot_feature = True
            break
    if is_onehot_feature:
        continue # Already added, move to next processed column

    # Check if this processed column is a remainder (numerical, engineered, or LE/OE)
    # and its original form (without 'remainder__') is in user's request
    if col_name_in_processed_df.startswith('remainder__'):
        original_col_name = col_name_in_processed_df.replace('remainder__', '')
        if original_col_name in important_cols_user_request_satisfaction:
            X_satisfaction_columns.append(col_name_in_processed_df)
    # Check if it's a column that was not transformed by CT (e.g., if it was already encoded before CT)
    # and is directly in the user's request. This handles cases like 'OverTime', 'TenureCategory' etc.
    elif col_name_in_processed_df in important_cols_user_request_satisfaction:
        X_satisfaction_columns.append(col_name_in_processed_df)

# Remove duplicates while preserving order (if any, though the logic should prevent)
X_satisfaction_columns = list(dict.fromkeys(X_satisfaction_columns))

X = df_processed[X_satisfaction_columns]
y = df_processed['remainder__JobSatisfaction'] # Target for job satisfaction prediction

# Store the exact feature names and their order for the Streamlit app
satisfaction_model_feature_names = X.columns.tolist()
print("\nFinal Features used for Job Satisfaction Model training (order is CRITICAL):")
print(satisfaction_model_feature_names)


# --- Split Data ---
# No stratify for regression tasks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets for Job Satisfaction Prediction.")

# --- Train and Save Satisfaction Models + Evaluate ---
print("\nTraining, saving, and evaluating Job Satisfaction models...")

satisfaction_models = {
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Linear Regression': LinearRegression()
}

evaluation_results_satisfaction = {}

for name, model in satisfaction_models.items():
    print(f"\n--- Training and Evaluating {name} (Job Satisfaction) ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the trained model
    model_filename = f'satisfaction_{name.lower().replace(" ", "_")}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"{name} model trained and saved as '{model_filename}'.")

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    evaluation_results_satisfaction[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }

    print(f"Metrics for {name}:\n")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# --- Save Satisfaction Model Feature Names ---
# This file will tell the Streamlit app exactly which features to prepare for satisfaction prediction
try:
    with open('satisfaction_model_features.pkl', 'wb') as f:
        pickle.dump(satisfaction_model_feature_names, f)
    print("Job Satisfaction model feature names saved as 'satisfaction_model_features.pkl'.")
except Exception as e:
    print(f"Error saving job satisfaction model feature names: {e}")

# --- Save Satisfaction Evaluation Results ---
try:
    with open('satisfaction_evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results_satisfaction, f)
    print("Job Satisfaction evaluation results saved as 'satisfaction_evaluation_results.pkl'.")
except Exception as e:
    print(f"Error saving job satisfaction evaluation results: {e}")


print("\nJob Satisfaction Model Training and Evaluation completed successfully!")

# --- Display Comparative Results ---
print("\n--- Job Satisfaction Model Comparison ---")
comparison_df_satisfaction = pd.DataFrame({
    'Model': [],
    'MAE': [],
    'MSE': [],
    'RMSE': [],
    'R2 Score': []
})

for name, metrics in evaluation_results_satisfaction.items():
    comparison_df_satisfaction.loc[len(comparison_df_satisfaction)] = [
        name,
        metrics['MAE'],
        metrics['MSE'],
        metrics['RMSE'],
        metrics['R2 Score']
    ]

print(comparison_df_satisfaction.set_index('Model').round(4))