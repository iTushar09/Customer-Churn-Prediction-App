# %% [markdown]
# **1. Importing the dependencies**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# %% [markdown]
# **2. Data Loading and Understanding**

# %%
# Re-import necessary packages due to code state reset
import pandas as pd

# Load the CSV data again after reset
df = pd.read_csv("C:/Users/Tushar Chaudhari/OneDrive/Documents/Visual Studio/mini project/final/ccpdata.csv")
df.head()

# %%
df.shape

# %%
df.head()

# %%
pd.set_option("display.max_columns", None)

# %%
df.head(2)

# %%
df.info()

# %%
# dropping customerID column as this is not required for modelling
df = df.drop(columns=["customerID"])

# %%
df.head(2)

# %%
df.columns

# %%
print(df["gender"].unique())

# %%
print(df["SeniorCitizen"].unique())

# %%
# printing the unique values in all the columns

numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in df.columns:
  if col not in numerical_features_list:
    print(col, df[col].unique())
    print("-"*50)

# %%
print(df.isnull().sum())

# %%
#df["TotalCharges"] = df["TotalCharges"].astype(float)

# %%
df[df["TotalCharges"]==" "]

# %%
len(df[df["TotalCharges"]==" "])

# %%
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})

# %%
df["TotalCharges"] = df["TotalCharges"].astype(float)

# %%
df.info()

# %%
# checking the class distribution of target column
print(df["Churn"].value_counts())

# %% [markdown]
# **Insights:**
# 1. Customer ID removed as it is not required for modelling
# 2. No mmissing values in the dataset
# 3. Missing values in the TotalCharges column were replaced with 0
# 4. Class imbalance identified in the target

# %% [markdown]
# **3. Exploratory Data Analysis (EDA)**

# %%
df.shape

# %%
df.columns

# %%
df.head(2)

# %%
df.describe()

# %% [markdown]
# **Numerical Features - Analysis**

# %% [markdown]
# Understand the distribution of teh numerical features

# %%
def plot_histogram(df, column_name):

  plt.figure(figsize=(5, 3))
  sns.histplot(df[column_name], kde=True)
  plt.title(f"Distribution of {column_name}")

  # calculate the mean and median values for the columns
  col_mean = df[column_name].mean()
  col_median = df[column_name].median()

  # add vertical lines for mean and median
  plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
  plt.axvline(col_median, color="green", linestyle="-", label="Median")

  plt.legend()

  plt.show()

# %%
plot_histogram(df, "tenure")

# %%
plot_histogram(df, "MonthlyCharges")

# %%
plot_histogram(df, "TotalCharges")

# %% [markdown]
# **Box plot for numerical features**

# %%
def plot_boxplot(df, column_name):

  plt.figure(figsize=(5, 3))
  sns.boxplot(y=df[column_name])
  plt.title(f"Box Plot of {column_name}")
  plt.ylabel(column_name)
  plt.show

# %%
plot_boxplot(df, "tenure")

# %%
plot_boxplot(df, "MonthlyCharges")

# %%
plot_boxplot(df, "TotalCharges")

# %% [markdown]
# **Correlation Heatmap for numerical columns**

# %%
# correlation matrix - heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# Categorical features - Analysis

# %%
df.columns

# %%
df.info()

# %% [markdown]
# Countplot for categorical columns

# %%
object_cols = df.select_dtypes(include="object").columns.to_list()

object_cols = ["SeniorCitizen"] + object_cols

for col in object_cols:
  plt.figure(figsize=(5, 3))
  sns.countplot(x=df[col])
  plt.title(f"Count Plot of {col}")
  plt.show()

# %% [markdown]
# **4. Data Preprocessing**

# %%
df.head(3)

# %% [markdown]
# Label encoding of target column

# %%
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

# %%
df.head(3)

# %%
print(df["Churn"].value_counts())

# %% [markdown]
# Label encoding of categorical fetaures

# %%
# identifying columns with object data type
object_columns = df.select_dtypes(include="object").columns

# %%
print(object_columns)

# %%
# initialize a dictionary to save the encoders
encoders = {}

# apply label encoding and store the encoders
for column in object_columns:
  label_encoder = LabelEncoder()
  df[column] = label_encoder.fit_transform(df[column])
  encoders[column] = label_encoder


# save the encoders to a pickle file
with open("encoders.pkl", "wb") as f:
  pickle.dump(encoders, f)


# %%
encoders

# %%
df.head()

# %% [markdown]
# **Traianing and test data split**

# %%
# splitting the features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# %%
# split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
print(y_train.shape)

# %%
print(y_train.value_counts())

# %%



# %% [markdown]
# **22app**

# %%




# %% [markdown]
# Synthetic Minority Oversampling TEchnique (SMOTE)

# %%
smote = SMOTE(random_state=42)

# %%
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# %%
print(y_train_smote.shape)

# %%
print(y_train_smote.value_counts())

# %% [markdown]
# **5. Model Training**

# %% [markdown]
# Training with default hyperparameters

# %%
# dictionary of models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# %%
# dictionary to store the cross validation results
cv_scores = {}

# perform 5-fold cross validation for each model
for model_name, model in models.items():
  print(f"Training {model_name} with default parameters")
  scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
  cv_scores[model_name] = scores
  print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")
  print("-"*70)

# %%
cv_scores

# %% [markdown]
# Random Forest gives the highest accuracy compared to other models with default parameters

# %%
rfc = RandomForestClassifier(random_state=42)

# %%
rfc.fit(X_train_smote, y_train_smote)

# %%
print(y_test.value_counts())

# %% [markdown]
# **6. Model Evaluation**

# %%
# evaluate on test data
y_test_pred = rfc.predict(X_test)

print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confsuion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# %%
# save the trained model as a pickle file
model_data = {"model": rfc, "features_names": X.columns.tolist()}


with open("customer_churn_model.pkl", "wb") as f:
  pickle.dump(model_data, f)

# %% [markdown]
# **7. Load the saved  model and  build a Predictive System**

# %%
# load teh saved model and the feature names

with open("customer_churn_model.pkl", "rb") as f:
  model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# %%
print(loaded_model)

# %%
print(feature_names)

# %%
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}


input_data_df = pd.DataFrame([input_data])

with open("encoders.pkl", "rb") as f:
  encoders = pickle.load(f)


# encode categorical featires using teh saved encoders
for column, encoder in encoders.items():
  input_data_df[column] = encoder.transform(input_data_df[column])

# make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(prediction)

# results
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediciton Probability: {pred_prob}")

# %%
from sklearn.preprocessing import LabelEncoder
import joblib

# Columns to encode
encode_cols = [
    "Contract", "PaymentMethod", "OnlineSecurity", "TechSupport",
    "InternetService", "OnlineBackup", "PaperlessBilling"
]

# Create a dictionary of encoders
encoders = {}

for col in encode_cols:
   le_contract = LabelEncoder()
df['Contract'] = le_contract.fit_transform(df['Contract'])  # This modifies df in-place
    encoders[col] = le  # Store for use later

# Now train your model on df with encoded columns...


encoders = {
    "Contract": le_contract,
    ...
}
joblib.dump(encoders, 'encoders.pkl')
# Save model and encoders
joblib.dump(model, "customer_churn_model.pkl")
joblib.dump(encoders, "encoders.pkl")


# %%
from sklearn.preprocessing import LabelEncoder
import joblib

# Categorical columns to encode
encode_cols = [
    "Contract", "PaymentMethod", "OnlineSecurity", "TechSupport",
    "InternetService", "OnlineBackup", "PaperlessBilling"
]

encoders = {}
for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# %%
# Make sure these match Streamlit options exactly
df["Contract"].unique()
df["PaymentMethod"].unique()
df["OnlineSecurity"].unique()
...


# %%
encoders = {}
for col in ["Contract", "PaymentMethod", "OnlineSecurity", "TechSupport", "InternetService", "OnlineBackup", "PaperlessBilling"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, 'encoders.pkl')


# %%
from sklearn.preprocessing import LabelEncoder
import joblib

# Initialize encoders
le_contract = LabelEncoder()
le_payment = LabelEncoder()
le_security = LabelEncoder()
le_support = LabelEncoder()
le_internet = LabelEncoder()
le_backup = LabelEncoder()
le_paperless = LabelEncoder()

# Fit and transform each relevant column
df['Contract'] = le_contract.fit_transform(df['Contract'])
df['PaymentMethod'] = le_payment.fit_transform(df['PaymentMethod'])
df['OnlineSecurity'] = le_security.fit_transform(df['OnlineSecurity'])
df['TechSupport'] = le_support.fit_transform(df['TechSupport'])
df['InternetService'] = le_internet.fit_transform(df['InternetService'])
df['OnlineBackup'] = le_backup.fit_transform(df['OnlineBackup'])
df['PaperlessBilling'] = le_paperless.fit_transform(df['PaperlessBilling'])

# Train your model here (X, y split, fit model, etc.)
# model = ...

# Save the trained model
joblib.dump(model, 'customer_churn_model.pkl')

# Save the encoders
encoders = {
    "Contract": le_contract,
    "PaymentMethod": le_payment,
    "OnlineSecurity": le_security,
    "TechSupport": le_support,
    "InternetService": le_internet,
    "OnlineBackup": le_backup,
    "PaperlessBilling": le_paperless
}
joblib.dump(encoders, 'encoders.pkl')



