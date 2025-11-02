# %%
import pandas as pd
import numpy as np

# Loading the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first 5 rows
print(df.head())

# %%
# Check shape and types
print("Shape:", df.shape)
print("Columns:\n", df.dtypes)

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Summary stats
df.describe(include='all')

# %%
# Convert TotalCharges to numeric (some have spaces or empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Dropping rows with missing TotalCharges
df.dropna(subset=['TotalCharges'], inplace=True)

# Confirm changes
print("Missing after cleanup:\n", df.isnull().sum())

# %%
# Dropping customerID which it is not needed here.
df.drop(['customerID'], axis=1, inplace=True)

# %%
# Converting target(Churn) column to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Check balance
df['Churn'].value_counts(normalize=True)

# %%
# checking which columns are object type (categorical features)
cat_cols = df.select_dtypes(include='object').columns
print("Categorical Columns:\n", cat_cols)


# %%
# Converting Churn to numeric (errors become NaN)
df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

# Dropping rows with missing Churn
df.dropna(subset=['Churn'], inplace=True)

df.reset_index(drop=True, inplace=True)

# One-hot encode all categorical variables

#This code converts all categorical columns into numeric dummy 
# variables (one-hot encoding) and avoids redundancy using 
# drop_first=True, preparing the dataset for machine learning
df_encoded = pd.get_dummies(df, drop_first=True)


# %%
## Check shape and preview
print("Encoded shape:", df_encoded.shape)

# %%
df_encoded.head()

# %%

#To normalize / scale numerical columns so they are on a similar range.

# Why?
# Because different features have different scales
#A machine learning model may give more importance to large-range values 
# like TotalCharges just because they are bigger numbers - not because
#  they matter more.

# 👉 Standardization fixes this issue ✅


from sklearn.preprocessing import StandardScaler

# Define numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Initialize and apply scaler
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# %%
#👉 This code divides customer churn data into training and testing sets,
#  preserving the churn class distribution, so the model can be trained 
# fairly and evaluated properly ✅
from sklearn.model_selection import train_test_split

# Separate features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# %%
#Train Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Train the model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict
y_pred_log = log_model.predict(X_test)

# %%
print(y_pred_log)

# %%
#Train Random Forest
from sklearn.ensemble import RandomForestClassifier

# Train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# %%
#Evaluate Both Models
#create a reusable function to evaluate both models
#his function saves time and keeps your model comparison clean, 
# structured, and easy to read.
def evaluate_model(name, y_test, y_pred):
    print(f"---- {name} ----")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 40)

# %%
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Random Forest", y_test, y_pred_rf)

# %%
#Plot ROC Curve
#The ROC Curve shows how confidently your model can separate churners 
# from non-churners, and AUC tells how good that separation is overall.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(model, X_test, y_test, label):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('How Good Are We at Spotting Ghosters?')
    plt.legend()

#%%
plot_roc_curve(log_model, X_test, y_test, "Logistic Regression")
plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
plt.show()

# %%
#Feature Importance (Random Forest)
#This code helps you understand which features (columns) are the most 
# influential in predicting customer churn.

# Basically:
# 👉 “Which customer attributes made the biggest difference in the model’s 
# decision to predict churn?”
import pandas as pd

# Get feature importances
importances = rf_model.feature_importances_
features = X.columns

# Create DataFrame
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Display top 10
feat_imp_df.head(10)

# %%
import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature')
plt.title('Top 10 Reasons Customers are Breaking Up with Us')
plt.show()

# %%
