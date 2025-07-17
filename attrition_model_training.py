import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting Employee Attrition Model Training and Evaluation...")

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


# --- Define Features (X) and Target (y) for Attrition Prediction ---
# These are the original column names and engineered features the user explicitly specified
# This list includes both numerical and original categorical features.
important_cols_user_request_attrition = [
    'JobSatisfaction', 'EnvironmentSatisfaction', 'JobInvolvement','WorkLifeBalance',
    'EngagementScore',  'YearsAtCompany', 'YearsPerCompany', 'PromotionGap',
    'ManagerTenureRatio', 'OverTime', 'MonthlyIncome', 'Age', 'TenureCategory', 'DailyRate',
    'JobLevel', 'StockOptionLevel', 'TotalWorkingYears', 'YearsInCurrentRole',
    'YearsWithCurrManager', 'YearsInSameRole', 'Gender', 'JobRole', 'MaritalStatus'
]

# Dynamically build the list of actual features that will be in X for attrition model.
# This ensures only the requested features (and their OHE forms) are included.
# The order will be based on how they appear in final_feature_names_all_processed_df,
# which is derived from the ColumnTransformer's consistent output order.
X_attrition_columns = []
# Get the original column names that were one-hot encoded by the CT
onehot_cols_from_ct = preprocessor_ct.named_transformers_['onehot'].feature_names_in_

for col_name_in_processed_df in final_feature_names_all_processed_df:
    # Check if this processed column is a one-hot encoded feature AND its original form is in user's request
    is_onehot_feature = False
    for original_cat_col in onehot_cols_from_ct:
        if col_name_in_processed_df.startswith(f'onehot__{original_cat_col}_') and original_cat_col in important_cols_user_request_attrition:
            X_attrition_columns.append(col_name_in_processed_df)
            is_onehot_feature = True
            break
    if is_onehot_feature:
        continue # Already added, move to next processed column

    # Check if this processed column is a remainder (numerical, engineered, or LE/OE)
    # and its original form (without 'remainder__') is in user's request
    if col_name_in_processed_df.startswith('remainder__'):
        original_col_name = col_name_in_processed_df.replace('remainder__', '')
        if original_col_name in important_cols_user_request_attrition:
            X_attrition_columns.append(col_name_in_processed_df)
    # Check if it's a column that was not transformed by CT (e.g., if it was already encoded before CT)
    # and is directly in the user's request. This handles cases like 'OverTime', 'TenureCategory' etc.
    elif col_name_in_processed_df in important_cols_user_request_attrition:
        X_attrition_columns.append(col_name_in_processed_df)


# Remove duplicates while preserving order (if any, though the logic should prevent)
X_attrition_columns = list(dict.fromkeys(X_attrition_columns))

X = df_processed[X_attrition_columns]
# CRITICAL FIX: Access 'Attrition' with its correct prefixed name
y = df_processed['remainder__Attrition']

# Store the exact feature names and their order for the Streamlit app
attrition_model_feature_names = X.columns.tolist()
print("\nFinal Features used for Attrition Model training (order is CRITICAL):")
print(attrition_model_feature_names)


# --- Split Data ---
# Use stratify for classification tasks to maintain class balance in splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nData split into training and testing sets for Attrition Prediction.")

# --- Train and Save Attrition Models + Evaluate ---
print("\nTraining, saving, and evaluating Attrition models...")

attrition_models = {
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
}

evaluation_results_attrition = {}

for name, model in attrition_models.items():
    print(f"\n--- Training and Evaluating {name} (Attrition) ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of attrition (class 1)

    # Save the trained model
    model_filename = f'attrition_{name.lower().replace(" ", "_")}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"{name} model trained and saved as '{model_filename}'.")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    evaluation_results_attrition[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'Confusion Matrix': conf_matrix
    }

    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Attrition', 'Predicted Attrition'],
                yticklabels=['Actual No Attrition', 'Actual Attrition'])
    plt.title(f'Confusion Matrix for {name} (Attrition)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# --- Save Attrition Model Feature Names ---
# This file will tell the Streamlit app exactly which features to prepare for attrition prediction
try:
    with open('attrition_model_features.pkl', 'wb') as f:
        pickle.dump(attrition_model_feature_names, f)
    print("Attrition model feature names saved as 'attrition_model_features.pkl'.")
except Exception as e:
    print(f"Error saving attrition model feature names: {e}")

# --- Save Attrition Evaluation Results ---
try:
    with open('attrition_evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results_attrition, f)
    print("Attrition evaluation results saved as 'attrition_evaluation_results.pkl'.")
except Exception as e:
    print(f"Error saving attrition evaluation results: {e}")


print("\nEmployee Attrition Model Training and Evaluation completed successfully!")

# --- Display Comparative Results ---
print("\n--- Attrition Model Comparison ---")
comparison_df_attrition = pd.DataFrame({
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'AUC-ROC': []
})

for name, metrics in evaluation_results_attrition.items():
    comparison_df_attrition.loc[len(comparison_df_attrition)] = [
        name,
        metrics['Accuracy'],
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1-Score'],
        metrics['AUC-ROC']
    ]

print(comparison_df_attrition.set_index('Model').round(4))

