import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("Starting Data Visualization and Analysis generation...")

# Create a directory for plots if it doesn't exist
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Load the raw dataset ---
try:
    df = pd.read_csv('Employee_Attrition.csv')
    print("Raw dataset loaded successfully for analysis.")
except FileNotFoundError:
    print("Error: 'Employee_Attrition.csv' not found. Please ensure it's in the same directory.")
    exit()

# --- KPI Calculations ---
print("\n--- Key Performance Indicators (KPIs) ---")

# 1. Overall Attrition Rate (Voluntary Attrition Rate, assuming all 'Yes' are voluntary for this dataset)
overall_attrition_rate = df['Attrition'].value_counts(normalize=True).get('Yes', 0) * 100
print(f"Overall (Voluntary) Attrition Rate: {overall_attrition_rate:.2f}%\n")

# 2. New Hire Attrition Rate (e.g., employees with 1 year or less at the company)
new_hires_df = df[df['YearsAtCompany'] <= 1]
new_hire_attrition_rate = new_hires_df['Attrition'].value_counts(normalize=True).get('Yes', 0) * 100
print(f"New Hire Attrition Rate (<= 1 year at company): {new_hire_attrition_rate:.2f}%\n")

# 3. Retention Rate
retention_rate = 100 - overall_attrition_rate
print(f"Overall Retention Rate: {retention_rate:.2f}%\n")

# 4. Average Tenure of Employees (using TotalWorkingYears and YearsAtCompany)
average_total_working_years = df['TotalWorkingYears'].mean()
average_years_at_company = df['YearsAtCompany'].mean()
print(f"Average Total Working Years: {average_total_working_years:.2f} years\n")
print(f"Average Years at Company: {average_years_at_company:.2f} years\n")

# 5. Average Job Satisfaction and Work-Life Balance
average_job_satisfaction = df['JobSatisfaction'].mean()
average_work_life_balance = df['WorkLifeBalance'].mean()
print(f"Average Job Satisfaction (1-4 scale): {average_job_satisfaction:.2f}\n")
print(f"Average Work-Life Balance (1-4 scale): {average_work_life_balance:.2f}\n")

# Store KPIs in a dictionary for easy access in Streamlit
kpis = {
    "Overall Attrition Rate": f"{overall_attrition_rate:.2f}%",
    "New Hire Attrition Rate (<= 1 year)": f"{new_hire_attrition_rate:.2f}%",
    "Overall Retention Rate": f"{retention_rate:.2f}%",
    "Average Total Working Years": f"{average_total_working_years:.2f} years",
    "Average Years at Company": f"{average_years_at_company:.2f} years",
    "Average Job Satisfaction (1-4 scale)": f"{average_job_satisfaction:.2f}",
    "Average Work-Life Balance (1-4 scale)": f"{average_work_life_balance:.2f}"
}
with open(os.path.join(PLOTS_DIR, 'kpis.pkl'), 'wb') as f:
    pickle.dump(kpis, f)
print(f"KPIs saved to {os.path.join(PLOTS_DIR, 'kpis.pkl')}")


# --- Visualizations ---
print("\nGenerating visualizations...")

# Set a consistent style for plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10}) # Adjust font size for better readability

# Figure 1: Overall Attrition, Gender, Department, JobRole, JobSatisfaction, WorkLifeBalance, MonthlyIncome, TotalWorkingYears
plt.figure(figsize=(20, 25))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 1. Overall Attrition Rate (Pie Chart)
plt.subplot(4, 2, 1)
attrition_counts = df['Attrition'].value_counts()
plt.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3b3','#ff9999'])
plt.title('Overall Attrition Rate')
plt.axis('equal')
plt.savefig(os.path.join(PLOTS_DIR, 'overall_attrition_pie.png'))

# 2. Attrition by Gender (Stacked Bar Chart)
plt.subplot(4, 2, 2)
attrition_by_gender = df.groupby('Gender')['Attrition'].value_counts(normalize=True).unstack() * 100
attrition_by_gender.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
plt.title('Attrition by Gender')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_by_gender.png'))

# 3. Attrition by Department (Stacked Bar Chart)
plt.subplot(4, 2, 3)
attrition_by_department = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack() * 100
attrition_by_department.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
plt.title('Attrition by Department')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_by_department.png'))

# 4. Attrition by JobRole (Stacked Bar Chart)
plt.subplot(4, 2, 4)
attrition_by_jobrole = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack() * 100
attrition_by_jobrole.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
plt.title('Attrition by Job Role')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_by_jobrole.png'))

# 5. Attrition by JobSatisfaction (Stacked Bar Chart)
plt.subplot(4, 2, 5)
attrition_by_jobsatisfaction = df.groupby('JobSatisfaction')['Attrition'].value_counts(normalize=True).unstack() * 100
attrition_by_jobsatisfaction.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
plt.title('Attrition by Job Satisfaction')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_by_jobsatisfaction.png'))

# 6. Attrition by WorkLifeBalance (Stacked Bar Chart)
plt.subplot(4, 2, 6)
attrition_by_worklifebalance = df.groupby('WorkLifeBalance')['Attrition'].value_counts(normalize=True).unstack() * 100
attrition_by_worklifebalance.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
plt.title('Attrition by Work-Life Balance')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_by_worklifebalance.png'))

# 7. Attrition by MonthlyIncome (Binning and Stacked Bar Chart)
df['MonthlyIncome_Bin'] = pd.cut(df['MonthlyIncome'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.subplot(4, 2, 7)
attrition_by_monthlyincome = df.groupby('MonthlyIncome_Bin')['Attrition'].value_counts(normalize=True).unstack() * 100
attrition_by_monthlyincome.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
plt.title('Attrition by Monthly Income')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_by_monthlyincome.png'))

# 8. Attrition by TotalWorkingYears (Binning and Stacked Bar Chart)
df['TotalWorkingYears_Bin'] = pd.cut(df['TotalWorkingYears'], bins=5, labels=['0-5', '6-10', '11-15', '16-20', '21+'])
plt.subplot(4, 2, 8)
attrition_by_totalworkingyears = df.groupby('TotalWorkingYears_Bin')['Attrition'].value_counts(normalize=True).unstack() * 100
attrition_by_totalworkingyears.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='Pastel1')
plt.title('Attrition by Total Working Years')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_by_totalworkingyears.png'))

plt.tight_layout()
# Do not plt.show() here, as Streamlit will display the images


# Figure 2: New KPIs Visualizations
plt.figure(figsize=(15, 6))
plt.subplots_adjust(wspace=0.3)

# Visualization for New Hire Attrition vs. Others
plt.subplot(1, 2, 1)
df['EmployeeCategory'] = df['YearsAtCompany'].apply(lambda x: 'New Hire (<=1 Year)' if x <= 1 else 'Experienced (>1 Year)')
new_hire_category_attrition = df.groupby('EmployeeCategory')['Attrition'].value_counts(normalize=True).unstack() * 100
new_hire_category_attrition.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('Attrition Rate: New Hires vs. Experienced Employees')
plt.ylabel('Percentage')
plt.xticks(rotation=0)
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'new_hire_attrition_vs_experienced.png'))

# Visualization for Distribution of Years At Company (Tenure)
plt.subplot(1, 2, 2)
sns.histplot(df['YearsAtCompany'], bins=10, kde=True, ax=plt.gca(), color='lightgreen')
plt.title('Distribution of Years At Company (Tenure)')
plt.xlabel('Years at Company')
plt.ylabel('Number of Employees')
plt.savefig(os.path.join(PLOTS_DIR, 'years_at_company_distribution.png'))

plt.tight_layout()


# Figure 3: Additional Attrition Analysis Visualizations
plt.figure(figsize=(20, 25))
plt.subplots_adjust(hspace=0.6, wspace=0.3)

# 1. Department-wise Average Monthly Income against Attrition
plt.subplot(4, 2, 1)
dept_attr_income = df.groupby(['Department', 'Attrition'])['MonthlyIncome'].mean().unstack()
dept_attr_income.plot(kind='bar', ax=plt.gca(), cmap='viridis')
plt.title('Department-wise Average Monthly Income vs Attrition')
plt.ylabel('Average Monthly Income')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'dept_attrition_income.png'))

# 2. Department-wise Average Monthly Income against Job Satisfaction
plt.subplot(4, 2, 2)
dept_jsat_income = df.groupby(['Department', 'JobSatisfaction'])['MonthlyIncome'].mean().unstack()
dept_jsat_income.plot(kind='bar', ax=plt.gca(), cmap='plasma')
plt.title('Department-wise Average Monthly Income vs Job Satisfaction')
plt.ylabel('Average Monthly Income')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Job Satisfaction')
plt.savefig(os.path.join(PLOTS_DIR, 'dept_jobsatisfaction_income.png'))

# 3. Job Role-wise Average Monthly Income against Attrition
plt.subplot(4, 2, 3)
job_attr_income = df.groupby(['JobRole', 'Attrition'])['MonthlyIncome'].mean().unstack()
job_attr_income.plot(kind='bar', ax=plt.gca(), cmap='viridis')
plt.title('Job Role-wise Average Monthly Income vs Attrition')
plt.ylabel('Average Monthly Income')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition')
plt.savefig(os.path.join(PLOTS_DIR, 'jobrole_attrition_income.png'))

# 4. Job Role-wise Average Monthly Income against Job Satisfaction
plt.subplot(4, 2, 4)
job_jsat_income = df.groupby(['JobRole', 'JobSatisfaction'])['MonthlyIncome'].mean().unstack()
job_jsat_income.plot(kind='bar', ax=plt.gca(), cmap='plasma')
plt.title('Job Role-wise Average Monthly Income vs Job Satisfaction')
plt.ylabel('Average Monthly Income')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Job Satisfaction')
plt.savefig(os.path.join(PLOTS_DIR, 'jobrole_jobsatisfaction_income.png'))

# 5. Department-wise Employee Count
plt.subplot(4, 2, 5)
department_counts = df['Department'].value_counts()
department_counts.plot(kind='bar', ax=plt.gca(), color='skyblue')
plt.title('Department-wise Employee Count')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45, ha='right')
plt.savefig(os.path.join(PLOTS_DIR, 'department_employee_count.png'))

# 6. Attrition Count by Department (from example)
plt.subplot(4, 2, 6)
attrition_yes = df[df['Attrition'] == 'Yes']
attrition_summary = attrition_yes.groupby('Department').agg(
    Headcount=('Attrition', 'count')
).reset_index()
sns.barplot(data=attrition_summary, x='Department', y='Headcount', palette="Reds", ax=plt.gca())
plt.title('Attrition Count by Department')
plt.ylabel('Employees with Attrition = Yes')
plt.xlabel('Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'attrition_count_by_department.png'))


plt.tight_layout()


# --- Model Comparison Results ---
print("\n--- Loading Model Comparison Results ---")

attrition_eval_path = 'attrition_evaluation_results.pkl'
satisfaction_eval_path = 'satisfaction_evaluation_results.pkl'

attrition_results = {}
satisfaction_results = {}

try:
    with open(attrition_eval_path, 'rb') as f:
        attrition_results = pickle.load(f)
    print(f"Attrition model evaluation results loaded from '{attrition_eval_path}'.")
except FileNotFoundError:
    print(f"Warning: Attrition evaluation results file '{attrition_eval_path}' not found. Please run attrition_model_training.py.")
except Exception as e:
    print(f"Error loading attrition evaluation results: {e}")

try:
    with open(satisfaction_eval_path, 'rb') as f:
        satisfaction_results = pickle.load(f)
    print(f"Job Satisfaction model evaluation results loaded from '{satisfaction_eval_path}'.")
except FileNotFoundError:
    print(f"Warning: Job Satisfaction evaluation results file '{satisfaction_eval_path}' not found. Please run satisfaction_model_training.py.")
except Exception as e:
    print(f"Error loading job satisfaction evaluation results: {e}")

# Prepare DataFrames for display in Streamlit
attrition_comparison_df = pd.DataFrame({
    'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'AUC-ROC': []
})
for name, metrics in attrition_results.items():
    attrition_comparison_df.loc[len(attrition_comparison_df)] = [
        name, metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score'], metrics['AUC-ROC']
    ]
attrition_comparison_df = attrition_comparison_df.set_index('Model').round(4)
print("\nAttrition Model Comparison Table:")
print(attrition_comparison_df.to_markdown())

satisfaction_comparison_df = pd.DataFrame({
    'Model': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2 Score': []
})
for name, metrics in satisfaction_results.items():
    satisfaction_comparison_df.loc[len(satisfaction_comparison_df)] = [
        name, metrics['MAE'], metrics['MSE'], metrics['RMSE'], metrics['R2 Score']
    ]
satisfaction_comparison_df = satisfaction_comparison_df.set_index('Model').round(4)
print("\nJob Satisfaction Model Comparison Table:")
print(satisfaction_comparison_df.to_markdown())

# Save comparison dataframes
try:
    with open(os.path.join(PLOTS_DIR, 'attrition_comparison_df.pkl'), 'wb') as f:
        pickle.dump(attrition_comparison_df, f)
    with open(os.path.join(PLOTS_DIR, 'satisfaction_comparison_df.pkl'), 'wb') as f:
        pickle.dump(satisfaction_comparison_df, f)
    print(f"Model comparison DataFrames saved to '{PLOTS_DIR}'.")
except Exception as e:
    print(f"Error saving comparison DataFrames: {e}")


print("\nData Visualization and Analysis generation completed. Plots saved to 'plots/' directory.")
