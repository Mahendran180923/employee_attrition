import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(page_title="Employee Analytics Dashboard", page_icon="�", layout="wide")

# --- Load Data and ML Resources ---
@st.cache_data
def load_raw_data(file_path):
    """Loads the raw Employee Attrition dataset for UI options and actual values."""
    if not os.path.exists(file_path):
        st.error(f"Error: Dataset file '{file_path}' not found. Please ensure it's in the same directory.")
        st.stop()
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading raw dataset: {e}")
        st.stop()

@st.cache_resource
def load_ml_resources():
    """Loads all pre-trained models, encoders, and feature lists."""
    resources = {}

    # Load all_fitted_encoders.pkl (contains ColumnTransformer and individual encoders)
    try:
        with open('all_fitted_encoders.pkl', 'rb') as f:
            resources['all_fitted_encoders'] = pickle.load(f)
        print("Encoders loaded successfully.") # Print to console instead of sidebar
    except Exception as e:
        st.sidebar.error(f"Error loading all_fitted_encoders.pkl: {e}")
        st.stop()

    # --- Load Best Attrition Model (Logistic Regression) ---
    attrition_model_path = 'attrition_logistic_regression.pkl' # Best model
    if not os.path.exists(attrition_model_path):
        st.sidebar.error(f"Best Attrition model '{attrition_model_path}' not found. Prediction will not work.")
        resources['best_attrition_model'] = None
    else:
        try:
            with open(attrition_model_path, 'rb') as f:
                resources['best_attrition_model'] = pickle.load(f)
            print("Best Attrition Model (Logistic Regression) loaded successfully.") # Print to console
        except Exception as e:
            st.sidebar.error(f"Error loading best attrition model: {e}")
            resources['best_attrition_model'] = None

    # Load Attrition Model Feature Names
    try:
        with open('attrition_model_features.pkl', 'rb') as f:
            resources['attrition_model_features'] = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading attrition_model_features.pkl: {e}")
        st.stop()

    # --- Load Best Satisfaction Model (Random Forest Regressor) ---
    satisfaction_model_path = 'satisfaction_random_forest_regressor.pkl' # Best model
    if not os.path.exists(satisfaction_model_path):
        st.sidebar.error(f"Best Satisfaction model '{satisfaction_model_path}' not found. Prediction will not work.")
        resources['best_satisfaction_model'] = None
    else:
        try:
            with open(satisfaction_model_path, 'rb') as f:
                resources['best_satisfaction_model'] = pickle.load(f)
            print("Best Satisfaction Model (Random Forest Regressor) loaded successfully.") # Print to console
        except Exception as e:
            st.sidebar.error(f"Error loading best satisfaction model: {e}")
            resources['best_satisfaction_model'] = None

    # Load Satisfaction Model Feature Names
    try:
        with open('satisfaction_model_features.pkl', 'rb') as f:
            resources['satisfaction_model_features'] = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"Error loading satisfaction_model_features.pkl: {e}")
        st.stop()

    # Load KPI and Model Comparison Data
    PLOTS_DIR = 'plots'
    # Initialize these to empty/default values
    resources['kpis'] = {}
    resources['attrition_comparison_df'] = pd.DataFrame()
    resources['satisfaction_comparison_df'] = pd.DataFrame()

    try:
        # Check if the plots directory exists before attempting to open files within it
        if os.path.exists(PLOTS_DIR):
            if os.path.exists(os.path.join(PLOTS_DIR, 'kpis.pkl')):
                with open(os.path.join(PLOTS_DIR, 'kpis.pkl'), 'rb') as f:
                    resources['kpis'] = pickle.load(f)
            else:
                print(f"KPIs file '{os.path.join(PLOTS_DIR, 'kpis.pkl')}' not found.") # Print to console

            if os.path.exists(os.path.join(PLOTS_DIR, 'attrition_comparison_df.pkl')):
                with open(os.path.join(PLOTS_DIR, 'attrition_comparison_df.pkl'), 'rb') as f:
                    resources['attrition_comparison_df'] = pickle.load(f)
            else:
                print(f"Attrition comparison file '{os.path.join(PLOTS_DIR, 'attrition_comparison_df.pkl')}' not found.") # Print to console

            if os.path.exists(os.path.join(PLOTS_DIR, 'satisfaction_comparison_df.pkl')):
                with open(os.path.join(PLOTS_DIR, 'satisfaction_comparison_df.pkl'), 'rb') as f:
                    resources['satisfaction_comparison_df'] = pickle.load(f)
            else:
                print(f"Satisfaction comparison file '{os.path.join(PLOTS_DIR, 'satisfaction_comparison_df.pkl')}' not found.") # Print to console

            # Only show "Analysis data loaded" if at least one of them loaded successfully
            if resources['kpis'] or not resources['attrition_comparison_df'].empty or not resources['satisfaction_comparison_df'].empty:
                print("Analysis data loaded successfully.") # Print to console
            else:
                st.sidebar.warning("No analysis data found. Please ensure 'data_visualization_analysis.py' ran successfully.")
        else:
            st.sidebar.warning(f"The 'plots/' directory was not found. Please run 'data_visualization_analysis.py' to generate analysis data and visualizations.")

    except Exception as e:
        st.sidebar.error(f"Error loading analysis data: {e}")


    return resources

# Load all resources at the start
df_raw = load_raw_data('Employee_Attrition.csv')
ml_resources = load_ml_resources()

# Initialize session state for autofill data
if 'autofill_data' not in st.session_state:
    st.session_state.autofill_data = None
if 'selected_employee_id' not in st.session_state:
    st.session_state.selected_employee_id = None

# Extract components for easier access
all_fitted_encoders = ml_resources['all_fitted_encoders']
best_attrition_model = ml_resources['best_attrition_model']
best_satisfaction_model = ml_resources['best_satisfaction_model']
attrition_model_features = ml_resources['attrition_model_features']
satisfaction_model_features = ml_resources['satisfaction_model_features']
kpis = ml_resources['kpis']
attrition_comparison_df = ml_resources['attrition_comparison_df']
satisfaction_comparison_df = ml_resources['satisfaction_comparison_df']


# --- Extract Unique Values for Categorical Features (for UI dropdowns) ---
categorical_cols_for_ui = {
    'Gender': df_raw['Gender'].unique().tolist(),
    'MaritalStatus': df_raw['MaritalStatus'].unique().tolist(),
    'JobRole': df_raw['JobRole'].unique().tolist(),
    'OverTime': df_raw['OverTime'].unique().tolist(),
    'BusinessTravel': df_raw['BusinessTravel'].unique().tolist(),
    'Department': df_raw['Department'].unique().tolist(),
    'EducationField': df_raw['EducationField'].unique().tolist(),
    # Ordinal order for TenureCategory (must match training)
    'TenureCategory_OrdinalOrder': ['<2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs']
}

# Get modes for non-UI original columns from df_raw for default values
default_values_non_ui = {
    'EmployeeCount': df_raw['EmployeeCount'].mode()[0],
    # 'EmployeeNumber': 1, # EmployeeNumber will be handled by selectbox or user input
    'Over18': df_raw['Over18'].mode()[0],
    'StandardHours': df_raw['StandardHours'].mode()[0]
}

# --- Feature Engineering Function (must match data_preparation.py) ---
def engineer_features(data):
    """Applies feature engineering logic to the input data."""
    df = data.copy()

    # Impute NumCompaniesWorked
    if df['NumCompaniesWorked'].iloc[0] == 0 and \
       (df['TotalWorkingYears'].iloc[0] - df['YearsAtCompany'].iloc[0]) == 1:
        df.loc[df.index, 'NumCompaniesWorked'] = 1

    # ManagerTenureRatio
    df['ManagerTenureRatio'] = df['YearsWithCurrManager'] / df['YearsAtCompany'].replace(0, np.nan)
    df['ManagerTenureRatio'] = df['ManagerTenureRatio'].fillna(0)

    # TenureCategory
    years_at_company_val = df['YearsAtCompany'].iloc[0]
    if years_at_company_val < 2:
        df['TenureCategory'] = '<2 yrs'
    elif 2 <= years_at_company_val <= 5:
        df['TenureCategory'] = '2-5 yrs'
    elif 5 < years_at_company_val <= 10:
        df['TenureCategory'] = '5-10 yrs'
    else:
        df['TenureCategory'] = '10+ yrs'

    # PerformanceMetric
    df['PerformanceMetric'] = df['PerformanceRating'] * df['PercentSalaryHike']

    # EngagementScore (JobSatisfaction removed from calculation as per previous fix)
    df['EngagementScore'] = (
        df['EnvironmentSatisfaction'] +
        df['RelationshipSatisfaction'] +
        df['JobInvolvement']
    )

    # PromotionGap
    df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']

    # YearsPerCompany
    df['YearsPerCompany'] = df['TotalWorkingYears'] / df['NumCompaniesWorked'] if df['NumCompaniesWorked'].iloc[0] > 0 else df['TotalWorkingYears']
    df['YearsPerCompany'] = df['YearsPerCompany'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # YearsSinceChange
    df['YearsSinceChange'] = df['YearsAtCompany'] - df['YearsInCurrentRole']

    # YearsInSameRole
    df['YearsInSameRole'] = np.where(
        df['YearsAtCompany'] == 0,
        0,
        df['YearsInCurrentRole'] / df['YearsAtCompany']
    )

    # TrainingPerYear
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

    return df

# --- Preprocessing Function for Prediction (applies encoders) ---
def preprocess_for_prediction(input_df_raw, target_type):
    """
    Applies encoding and prepares the DataFrame for prediction.
    Ensures columns match the trained model's expected features.
    """
    processed_input_df = input_df_raw.copy()

    # Apply individual LabelEncoders/OrdinalEncoders
    le_oe_cols = ['OverTime', 'Over18', 'BusinessTravel', 'TenureCategory']
    for col in le_oe_cols:
        if col in processed_input_df.columns and col in all_fitted_encoders:
            try:
                encoder_obj = all_fitted_encoders[col]
                if isinstance(encoder_obj, OrdinalEncoder):
                    processed_input_df[col] = encoder_obj.transform(processed_input_df[[col]])
                elif isinstance(encoder_obj, LabelEncoder):
                    processed_input_df[col] = encoder_obj.transform(processed_input_df[col])
            except ValueError as e:
                st.warning(f"Could not encode '{col}' with value '{processed_input_df[col].iloc[0]}'. Error: {e}. Setting to NaN.")
                processed_input_df[col] = np.nan
            except Exception as e:
                st.error(f"Error applying encoder for '{col}': {e}")
                st.stop()
        elif col in processed_input_df.columns:
            st.warning(f"Encoder for '{col}' not found. This column will not be encoded.")


    # Apply ColumnTransformer for OneHotEncoding
    if 'ColumnTransformer' in all_fitted_encoders:
        preprocessor_ct = all_fitted_encoders['ColumnTransformer']
        onehot_cols_ct = preprocessor_ct.named_transformers_['onehot'].feature_names_in_.tolist()
        passthrough_cols_ct = [col for col in processed_input_df.columns if col not in onehot_cols_ct]
        df_for_ct_ordered = processed_input_df[onehot_cols_ct + passthrough_cols_ct]

        try:
            transformed_array = preprocessor_ct.transform(df_for_ct_ordered)
            final_processed_cols = preprocessor_ct.get_feature_names_out()
            processed_input_df = pd.DataFrame(transformed_array, columns=final_processed_cols)

        except ValueError as e:
            st.error(f"Error during ColumnTransformer transformation: {e}. "
                     "This often happens if categories are new or columns are missing/misordered.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error with ColumnTransformer: {e}")
            st.stop()
    else:
        st.warning("ColumnTransformer not found in encoders. One-hot encoding will be skipped.")


    # Ensure all columns are numeric after encoding
    for col in processed_input_df.columns:
        if processed_input_df[col].dtype == 'object':
            processed_input_df[col] = pd.to_numeric(processed_input_df[col], errors='coerce')
            processed_input_df[col] = processed_input_df[col].fillna(-1)
        if processed_input_df[col].isnull().any():
            processed_input_df[col] = processed_input_df[col].fillna(processed_input_df[col].mean())

    # Select and reorder columns to match the specific model's expected features
    if target_type == 'attrition':
        expected_features = attrition_model_features
    elif target_type == 'satisfaction':
        expected_features = satisfaction_model_features
    else:
        st.error("Invalid target type specified for preprocessing.")
        st.stop()

    missing_cols = set(expected_features) - set(processed_input_df.columns)
    if missing_cols:
        for col in missing_cols:
            st.warning(f"Missing expected feature '{col}'. Adding with value 0.")
            processed_input_df[col] = 0

    extra_cols = set(processed_input_df.columns) - set(expected_features)
    if extra_cols:
        processed_input_df = processed_input_df.drop(columns=list(extra_cols))

    final_df_for_prediction = processed_input_df[expected_features]

    return final_df_for_prediction


# --- Main Application Layout ---
st.title("📊 Employee Analytics Dashboard")
st.markdown("---")

# Sidebar for navigation and model selection
st.sidebar.header("Navigation & Model Selection")
app_mode = st.sidebar.radio(
    "Go to",
    ("Prediction", "Analysis Dashboard")
)

if app_mode == "Prediction":
    prediction_type = st.sidebar.radio(
        "Choose Prediction Type:",
        ("Employee Attrition", "Job Satisfaction")
    )

    # --- Employee Selection and Autofill ---
    st.header("Employee Details for Prediction")

    # Dropdown for Employee Number
    employee_numbers = ['Manual Input'] + df_raw['EmployeeNumber'].unique().tolist()
    selected_employee_number = st.selectbox(
        "Select Employee Number (or choose Manual Input):",
        employee_numbers,
        index=0, # Default to Manual Input
        key="employee_num_selector"
    )

    # Button to trigger autofill
    autofill_button_clicked = False
    if selected_employee_number != 'Manual Input':
        if st.button(f"Load Data for Employee {selected_employee_number}"):
            st.session_state.selected_employee_id = selected_employee_number
            st.session_state.autofill_data = df_raw[df_raw['EmployeeNumber'] == selected_employee_number].iloc[0].to_dict()
            autofill_button_clicked = True
            st.success(f"Data for Employee {selected_employee_number} loaded. You can now modify if needed.")
        else:
            # If the selectbox changes but button isn't clicked, clear autofill
            if st.session_state.selected_employee_id != selected_employee_number:
                st.session_state.autofill_data = None
                st.session_state.selected_employee_id = None

    # Clear autofill if manual input is selected
    if selected_employee_number == 'Manual Input' and st.session_state.autofill_data is not None:
        st.session_state.autofill_data = None
        st.session_state.selected_employee_id = None


    # Helper to get initial value for input fields
    def get_initial_value(column_name, default_val, options=None):
        if st.session_state.autofill_data and column_name in st.session_state.autofill_data:
            autofilled_val = st.session_state.autofill_data[column_name]
            if options: # For selectbox, find index
                try:
                    return options.index(autofilled_val)
                except ValueError:
                    # If autofilled value is not in current options, fallback to default index (0)
                    return 0
            return autofilled_val # This path is for non-selectbox inputs, where autofilled_val is the direct value
        if options: # If no autofill data, but it's a selectbox, return index of default_val
            try:
                return options.index(default_val)
            except ValueError:
                return 0 # Fallback if default_val not in options
        return default_val # For non-selectbox, return default_val directly

    col1, col2, col3 = st.columns(3)

    # Define options for numerical selectboxes once
    satisfaction_options = [1, 2, 3, 4]
    education_options = [1, 2, 3, 4, 5]
    stock_options = [0, 1, 2, 3]


    with col1:
        st.subheader("Personal & Job Info")
        age = st.number_input("Age", 18, 65, value=int(get_initial_value('Age', 30)))
        gender = st.selectbox("Gender", categorical_cols_for_ui['Gender'], index=get_initial_value('Gender', categorical_cols_for_ui['Gender'][0], categorical_cols_for_ui['Gender']))
        marital_status = st.selectbox("Marital Status", categorical_cols_for_ui['MaritalStatus'], index=get_initial_value('MaritalStatus', categorical_cols_for_ui['MaritalStatus'][0], categorical_cols_for_ui['MaritalStatus']))
        job_role = st.selectbox("Job Role", categorical_cols_for_ui['JobRole'], index=get_initial_value('JobRole', categorical_cols_for_ui['JobRole'][0], categorical_cols_for_ui['JobRole']))
        job_level = st.number_input("Job Level", 1, 5, value=int(get_initial_value('JobLevel', 2)))
        over_time = st.selectbox("Over Time", categorical_cols_for_ui['OverTime'], index=get_initial_value('OverTime', categorical_cols_for_ui['OverTime'][0], categorical_cols_for_ui['OverTime']))
        business_travel = st.selectbox("Business Travel", categorical_cols_for_ui['BusinessTravel'], index=get_initial_value('BusinessTravel', categorical_cols_for_ui['BusinessTravel'][0], categorical_cols_for_ui['BusinessTravel']))
        department = st.selectbox("Department", categorical_cols_for_ui['Department'], index=get_initial_value('Department', categorical_cols_for_ui['Department'][0], categorical_cols_for_ui['Department']))
        education_field = st.selectbox("Education Field", categorical_cols_for_ui['EducationField'], index=get_initial_value('EducationField', categorical_cols_for_ui['EducationField'][0], categorical_cols_for_ui['EducationField']))


    with col2:
        st.subheader("Compensation & Satisfaction")
        performance_rating = st.selectbox("Performance Rating", satisfaction_options, index=get_initial_value('PerformanceRating', 3, satisfaction_options)) # Default to 3 (index 2)
        job_satisfaction_input = st.selectbox("Job Satisfaction (for Engagement Score)", satisfaction_options, index=get_initial_value('JobSatisfaction', 3, satisfaction_options)) # Default to 3 (index 2)
        environment_satisfaction = st.selectbox("Environment Satisfaction", satisfaction_options, index=get_initial_value('EnvironmentSatisfaction', 3, satisfaction_options)) # Default to 3 (index 2)
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", satisfaction_options, index=get_initial_value('RelationshipSatisfaction', 3, satisfaction_options)) # Default to 3 (index 2)
        job_involvement = st.selectbox("Job Involvement", satisfaction_options, index=get_initial_value('JobInvolvement', 3, satisfaction_options)) # Default to 3 (index 2)
        work_life_balance = st.selectbox("Work Life Balance", satisfaction_options, index=get_initial_value('WorkLifeBalance', 3, satisfaction_options)) # Default to 3 (index 2)
        stock_option_level = st.selectbox("Stock Option Level", stock_options, index=get_initial_value('StockOptionLevel', 1, stock_options)) # Default to 1 (index 1)
        daily_rate = st.number_input("Daily Rate", 100, 1500, value=int(get_initial_value('DailyRate', 800)))
        hourly_rate = st.number_input("Hourly Rate", 30, 100, value=int(get_initial_value('HourlyRate', 65)))
        monthly_income = st.number_input("Monthly Income", 1000, 20000, value=int(get_initial_value('MonthlyIncome', 6500)))
        monthly_rate = st.number_input("Monthly Rate", 2000, 27000, value=int(get_initial_value('MonthlyRate', 14000)))
        percent_salary_hike = st.number_input("Percent Salary Hike", 11, 25, value=int(get_initial_value('PercentSalaryHike', 15)))


    with col3:
        st.subheader("Experience & Tenure")
        education = st.selectbox("Education", education_options, index=get_initial_value('Education', 3, education_options)) # Default to 3 (index 2)
        distance_from_home = st.number_input("Distance From Home (miles)", 1, 29, value=int(get_initial_value('DistanceFromHome', 10)))
        num_companies_worked = st.number_input("Num Companies Worked", 0, 9, value=int(get_initial_value('NumCompaniesWorked', 2)))
        total_working_years = st.number_input("Total Working Years", 0, 40, value=int(get_initial_value('TotalWorkingYears', 10)))
        training_times_last_year = st.number_input("Training Times Last Year", 0, 6, value=int(get_initial_value('TrainingTimesLastYear', 2)))
        years_at_company = st.number_input("Years At Company", 0, 40, value=int(get_initial_value('YearsAtCompany', 5)))
        years_in_current_role = st.number_input("Years In Current Role", 0, 18, value=int(get_initial_value('YearsInCurrentRole', 3)))
        years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15, value=int(get_initial_value('YearsSinceLastPromotion', 1)))
        years_with_curr_manager = st.number_input("Years With Current Manager", 0, 17, value=int(get_initial_value('YearsWithCurrManager', 2)))


    # --- Prediction Logic ---
    if st.button("Get Prediction"):
        # Create a DataFrame from user inputs, including default values for non-UI columns
        input_data = {
            'Age': [age],
            'DailyRate': [daily_rate],
            'DistanceFromHome': [distance_from_home],
            'Education': [education],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'Gender': [gender],
            'HourlyRate': [hourly_rate],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobRole': [job_role],
            'JobSatisfaction': [job_satisfaction_input], # Use the input for EngagementScore calculation
            'MaritalStatus': [marital_status],
            'MonthlyIncome': [monthly_income],
            'MonthlyRate': [monthly_rate],
            'NumCompaniesWorked': [num_companies_worked],
            'OverTime': [over_time],
            'PercentSalaryHike': [percent_salary_hike],
            'PerformanceRating': [performance_rating],
            'RelationshipSatisfaction': [relationship_satisfaction],
            'StockOptionLevel': [stock_option_level],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_times_last_year],
            'WorkLifeBalance': [work_life_balance],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'YearsWithCurrManager': [years_with_curr_manager],
            # Add non-UI columns with their default values
            'BusinessTravel': [business_travel],
            'Department': [department],
            'EducationField': [education_field],
            'EmployeeCount': [default_values_non_ui['EmployeeCount']],
            'EmployeeNumber': [selected_employee_number if selected_employee_number != 'Manual Input' else default_values_non_ui['EmployeeNumber']],
            'Over18': [default_values_non_ui['Over18']],
            'StandardHours': [default_values_non_ui['StandardHours']],
            # CRITICAL FIX: Add Attrition with a placeholder value
            'Attrition': [0] # Placeholder, as CT expects this column
        }
        user_input_df_raw = pd.DataFrame(input_data)

        # Apply feature engineering
        engineered_df = engineer_features(user_input_df_raw)

        st.subheader("Prediction Result")

        # Display Actual Values if an employee was selected
        actual_attrition = None
        actual_job_satisfaction = None
        if st.session_state.selected_employee_id is not None:
            employee_actual_data = df_raw[df_raw['EmployeeNumber'] == st.session_state.selected_employee_id].iloc[0]
            actual_attrition = employee_actual_data['Attrition']
            actual_job_satisfaction = employee_actual_data['JobSatisfaction']
            st.markdown(f"### Actual Values for Employee {st.session_state.selected_employee_id}:")
            st.write(f"**Actual Attrition:** {actual_attrition}")
            st.write(f"**Actual Job Satisfaction:** {actual_job_satisfaction}")
            st.markdown("---")


        if prediction_type == "Employee Attrition":
            if best_attrition_model is None:
                st.error("Best Attrition model not loaded. Please check your model files.")
            else:
                # Preprocess for attrition prediction
                final_input_df = preprocess_for_prediction(engineered_df, 'attrition')

                try:
                    prediction = best_attrition_model.predict(final_input_df)
                    prediction_proba = best_attrition_model.predict_proba(final_input_df)[:, 1] # Probability of attrition (class 1)

                    if prediction[0] == 1:
                        st.error(f"**Predicted Attrition: High Likelihood of Attrition** 😟")
                        st.write(f"Probability of Attrition: **{prediction_proba[0]:.2f}**")
                        st.markdown("Consider reviewing this employee's situation to understand potential factors contributing to attrition.")
                    else:
                        st.success(f"**Predicted Attrition: Low Likelihood of Attrition** 😊")
                        st.write(f"Probability of Attrition: **{prediction_proba[0]:.2f}**")
                        st.markdown("This employee is likely to remain with the company.")
                except Exception as e:
                    st.error(f"Error during Attrition prediction: {e}")
                    st.write("Debug DataFrame for Attrition Prediction:")
                    st.write(final_input_df)
                    st.write("Expected features:")
                    st.write(attrition_model_features)

        elif prediction_type == "Job Satisfaction":
            if best_satisfaction_model is None:
                st.error("Best Job Satisfaction model not loaded. Please check your model files.")
            else:
                # Preprocess for job satisfaction prediction
                final_input_df = preprocess_for_prediction(engineered_df, 'satisfaction')

                try:
                    prediction = best_satisfaction_model.predict(final_input_df)
                    st.info(f"**Predicted Job Satisfaction Score: {prediction[0]:.2f}**")
                    st.markdown("*(Score is on a scale of 1 to 4, where 4 is highly satisfied)*")
                    if prediction[0] < 2.5:
                        st.warning("This employee's predicted job satisfaction is relatively low. Consider exploring factors that could improve their satisfaction.")
                    else:
                        st.success("This employee's predicted job satisfaction is relatively high.")

                except Exception as e:
                    st.error(f"Error during Job Satisfaction prediction: {e}")
                    st.write("Debug DataFrame for Job Satisfaction Prediction:")
                    st.write(final_input_df)
                    st.write("Expected features:")
                    st.write(satisfaction_model_features)

elif app_mode == "Analysis Dashboard":
    st.header("Employee Analytics Dashboard")
    st.markdown("This section provides key insights and model performance comparisons.")

    # --- Section: Key Performance Indicators (KPIs) ---
    with st.expander("Key Performance Indicators (KPIs)", expanded=True):
        if kpis:
            kpi_cols = st.columns(3)
            kpi_index = 0
            for kpi_name, kpi_value in kpis.items():
                with kpi_cols[kpi_index % 3]:
                    st.metric(label=kpi_name, value=kpi_value)
                kpi_index += 1
        else:
            st.warning("KPIs not loaded. Please run 'data_visualization_analysis.py'.")

    # --- Section: Model Performance Comparison ---
    with st.expander("Model Performance Comparison", expanded=False):
        st.markdown("#### Attrition Prediction Models")
        if not attrition_comparison_df.empty:
            st.dataframe(attrition_comparison_df)
        else:
            st.warning("Attrition model comparison data not loaded. Please run 'data_visualization_analysis.py'.")

        st.markdown("#### Job Satisfaction Prediction Models")
        if not satisfaction_comparison_df.empty:
            st.dataframe(satisfaction_comparison_df)
        else:
            st.warning("Job Satisfaction model comparison data not loaded. Please run 'data_visualization_analysis.py'.")

    # --- Section: Key Visualizations ---
    with st.expander("Key Visualizations", expanded=False):
        # Use a copy of df_raw for plotting to avoid modifying the cached original DataFrame
        df_plot = df_raw.copy()

        # Set a consistent style for plots
        # sns.set_style("whitegrid") # No longer needed if all Matplotlib plots are removed
        # plt.rcParams.update({'font.size': 10}) # No longer needed if all Matplotlib plots are removed

        st.markdown("##### Attrition Rate by Categories")

        col_viz_A1, col_viz_A2 = st.columns(2)
        with col_viz_A1:
            # --- Plot 1: Overall Attrition Rate (Pie Chart) - Plotly Express ---
            attrition_counts = df_plot['Attrition'].value_counts().reset_index()
            attrition_counts.columns = ['Attrition', 'Count']
            fig1_px = px.pie(
                attrition_counts,
                values='Count',
                names='Attrition',
                title='Overall Attrition Rate',
                color_discrete_sequence=['#66b3b3','#ff9999'] # Consistent colors
            )
            fig1_px.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1_px, use_container_width=True)

            # --- Plot 2: Attrition by Gender (Stacked Bar Chart) - Plotly Express ---
            gender_attrition_counts = df_plot.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
            gender_attrition_total = gender_attrition_counts.groupby('Gender')['Count'].transform('sum')
            gender_attrition_counts['Percentage'] = (gender_attrition_counts['Count'] / gender_attrition_total) * 100

            fig2_px = px.bar(
                gender_attrition_counts,
                x='Gender',
                y='Percentage',
                color='Attrition',
                title='Attrition by Gender',
                labels={'Percentage': 'Percentage of Employees'},
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}
            )
            fig2_px.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig2_px, use_container_width=True)

        with col_viz_A2:
            # --- Plot 3: Attrition by Department (Stacked Bar Chart) - Plotly Express ---
            dept_attrition_counts = df_plot.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
            dept_attrition_total = dept_attrition_counts.groupby('Department')['Count'].transform('sum')
            dept_attrition_counts['Percentage'] = (dept_attrition_counts['Count'] / dept_attrition_total) * 100

            fig3_px = px.bar(
                dept_attrition_counts,
                x='Department',
                y='Percentage',
                color='Attrition',
                title='Attrition by Department',
                labels={'Percentage': 'Percentage of Employees'},
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}
            )
            fig3_px.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig3_px, use_container_width=True)

            # --- Plot 4: Attrition by Job Role (Stacked Bar Chart) - Plotly Express ---
            jobrole_attrition_counts = df_plot.groupby(['JobRole', 'Attrition']).size().reset_index(name='Count')
            jobrole_attrition_total = jobrole_attrition_counts.groupby('JobRole')['Count'].transform('sum')
            jobrole_attrition_counts['Percentage'] = (jobrole_attrition_counts['Count'] / jobrole_attrition_total) * 100

            fig4_px = px.bar(
                jobrole_attrition_counts,
                x='JobRole',
                y='Percentage',
                color='Attrition',
                title='Attrition by Job Role',
                labels={'Percentage': 'Percentage of Employees'},
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}
            )
            fig4_px.update_layout(yaxis_range=[0, 100], xaxis_tickangle=-45) # Rotate labels for readability
            st.plotly_chart(fig4_px, use_container_width=True)

        st.markdown("##### Attrition Distribution")

        col_viz_B1, col_viz_B2 = st.columns(2)
        with col_viz_B1:
            # --- Plot 5: Attrition by Age Group (Histogram/Bar Chart) - Plotly Express ---
            df_plot['AgeGroup'] = pd.cut(df_plot['Age'], bins=[18, 25, 35, 45, 55, 65],
                                         labels=['18-24', '25-34', '35-44', '45-54', '55-64'], right=False)
            
            fig5_px = px.histogram(
                df_plot,
                x='AgeGroup',
                color='Attrition',
                title='Attrition by Age Group',
                category_orders={"AgeGroup": ['18-24', '25-34', '35-44', '45-54', '55-64']},
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'},
                barmode='group'
            )
            st.plotly_chart(fig5_px, use_container_width=True)

            # --- Plot 6: Attrition by Marital Status (Pie Chart) - Plotly Express ---
            marital_status_attrition = df_plot.groupby(['MaritalStatus', 'Attrition']).size().reset_index(name='Count')
            
            fig6_px = px.pie(
                marital_status_attrition,
                values='Count',
                names='MaritalStatus', # Main segments are MaritalStatus
                color='MaritalStatus', # Color by MaritalStatus for distinct slice colors
                hover_data=['Attrition', 'Count'], # Show Attrition info on hover
                title='Attrition by Marital Status',
                # color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}, # This is now less relevant for main slices
                hole=0.3 # Donut chart for better aesthetics
            )
            fig6_px.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig6_px, use_container_width=True)

        with col_viz_B2:
            # --- Plot 7: Attrition by OverTime (Pie Chart) - Plotly Express ---
            overtime_attrition = df_plot.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')

            fig7_px = px.pie(
                overtime_attrition,
                values='Count',
                names='OverTime', # Main segments are OverTime
                color='OverTime', # Color by OverTime for distinct slice colors
                hover_data=['Attrition', 'Count'], # Show Attrition info on hover
                title='Attrition by OverTime',
                # color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}, # Less relevant for main slices
                hole=0.3
            )
            fig7_px.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig7_px, use_container_width=True)

            # --- Plot 8: Monthly Income Distribution by Attrition (Histogram) - Plotly Express ---
            fig8_px = px.histogram(
                df_plot,
                x='MonthlyIncome',
                color='Attrition',
                title='Monthly Income Distribution by Attrition',
                marginal='box', # Add box plot on the margin
                nbins=50, # Number of bins
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}
            )
            fig8_px.update_layout(xaxis_title="Monthly Income")
            st.plotly_chart(fig8_px, use_container_width=True)

        st.markdown("##### Other Key Distributions")
        col_viz_C1, col_viz_C2 = st.columns(2)

        with col_viz_C1:
            # --- Plot 9: Distance From Home Distribution by Attrition (Histogram) - Plotly Express ---
            fig9_px = px.histogram(
                df_plot,
                x='DistanceFromHome',
                color='Attrition',
                title='Distance From Home Distribution by Attrition',
                marginal='box',
                nbins=20,
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}
            )
            fig9_px.update_layout(xaxis_title="Distance From Home (miles)")
            st.plotly_chart(fig9_px, use_container_width=True)

        with col_viz_C2:
            # --- Plot 10: YearsAtCompany vs. MonthlyIncome by Attrition (Scatter Plot) - Plotly Express ---
            fig10_px = px.scatter(
                df_plot,
                x='YearsAtCompany',
                y='MonthlyIncome',
                color='Attrition',
                hover_data=['JobRole', 'Age'], # Show additional info on hover
                title='Years At Company vs. Monthly Income by Attrition',
                labels={'YearsAtCompany': 'Years At Company', 'MonthlyIncome': 'Monthly Income'},
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'},
                template='plotly_white' # Use a clean template
            )
            st.plotly_chart(fig10_px, use_container_width=True)
        
        st.markdown("##### Satisfaction and Involvement")
        col_viz_D1, col_viz_D2 = st.columns(2)

        with col_viz_D1:
            # --- Plot 11: Job Satisfaction Distribution (Bar Chart) - Plotly Express ---
            job_satisfaction_counts = df_plot['JobSatisfaction'].value_counts().sort_index().reset_index()
            job_satisfaction_counts.columns = ['JobSatisfaction', 'Count']
            
            fig11_px = px.bar(
                job_satisfaction_counts,
                x='JobSatisfaction',
                y='Count',
                title='Job Satisfaction Distribution',
                labels={'JobSatisfaction': 'Job Satisfaction Level (1-4)', 'Count': 'Number of Employees'},
                color='JobSatisfaction',
                color_continuous_scale=px.colors.sequential.Viridis # Use Viridis for numerical scale
            )
            fig11_px.update_layout(xaxis=dict(tickmode='array', tickvals=[1,2,3,4])) # Ensure specific ticks
            st.plotly_chart(fig11_px, use_container_width=True)

        with col_viz_D2:
            # --- Plot 12: Environment Satisfaction Distribution (Bar Chart) - Plotly Express ---
            env_satisfaction_counts = df_plot['EnvironmentSatisfaction'].value_counts().sort_index().reset_index()
            env_satisfaction_counts.columns = ['EnvironmentSatisfaction', 'Count']

            fig12_px = px.bar(
                env_satisfaction_counts,
                x='EnvironmentSatisfaction',
                y='Count',
                title='Environment Satisfaction Distribution',
                labels={'EnvironmentSatisfaction': 'Environment Satisfaction Level (1-4)', 'Count': 'Number of Employees'},
                color='EnvironmentSatisfaction',
                color_continuous_scale=px.colors.sequential.Plasma # Using Plasma colormap
            )
            fig12_px.update_layout(xaxis=dict(tickmode='array', tickvals=[1,2,3,4]))
            st.plotly_chart(fig12_px, use_container_width=True)

        st.markdown("##### Income & Job Role Insights")
        col_viz_E1, col_viz_E2 = st.columns(2)

        with col_viz_E1:
            # --- Plot 13: Job Role-wise Average Monthly Income against Attrition (Bar Chart) - Plotly Express ---
            job_attr_income = df_plot.groupby(['JobRole', 'Attrition'])['MonthlyIncome'].mean().reset_index()
            
            fig13_px = px.bar(
                job_attr_income,
                x='JobRole',
                y='MonthlyIncome',
                color='Attrition',
                barmode='group',
                title='Job Role-wise Average Monthly Income vs Attrition',
                labels={'MonthlyIncome': 'Average Monthly Income'},
                color_discrete_map={'Yes': '#ff9999', 'No': '#66b3b3'}
            )
            fig13_px.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig13_px, use_container_width=True)

        with col_viz_E2:
            # --- Plot 14: Job Role-wise Average Monthly Income against Job Satisfaction (Bar Chart) - Plotly Express ---
            # To plot against Job Satisfaction, we need to consider JobSatisfaction as a continuous variable or group it.
            # Assuming you want average income for each JobRole, colored by average JobSatisfaction, or grouped by JS level.
            # If JobSatisfaction is a discrete number (1,2,3,4), it's best treated as a category for grouping.
            
            # Let's group by JobRole and JobSatisfaction and take the mean MonthlyIncome
            job_jsat_income = df_plot.groupby(['JobRole', 'JobSatisfaction'])['MonthlyIncome'].mean().reset_index()

            fig14_px = px.bar(
                job_jsat_income,
                x='JobRole',
                y='MonthlyIncome',
                color='JobSatisfaction', # Color by JobSatisfaction level
                barmode='group',
                title='Job Role-wise Average Monthly Income vs Job Satisfaction',
                labels={'MonthlyIncome': 'Average Monthly Income', 'JobSatisfaction': 'Job Satisfaction Level'},
                color_continuous_scale=px.colors.sequential.Plasma # Use plasma colormap for satisfaction levels
            )
            fig14_px.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig14_px, use_container_width=True)


st.markdown("---")
st.markdown("Developed for Employee Attrition and Job Satisfaction Prediction Project.")


