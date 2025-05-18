import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Sample synthetic dataset (replace with your real data)
data = pd.DataFrame({
    'tenure': np.random.randint(0, 72, 1000),
    'MonthlyCharges': np.random.uniform(20, 120, 1000),
    'TotalCharges': np.random.uniform(20, 8000, 1000),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 1000),
    'OnlineSecurity': np.random.choice(['No', 'Yes', 'No internet service'], 1000),
    'TechSupport': np.random.choice(['No', 'Yes', 'No internet service'], 1000),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000),
    'OnlineBackup': np.random.choice(['No', 'Yes', 'No internet service'], 1000),
    'PaperlessBilling': np.random.choice(['No', 'Yes'], 1000),
    'Churn': np.random.choice([0,1], 1000)
})

# Separate features and target
X = data.drop(columns=['Churn'])
y = data['Churn']

# Encode categorical variables with LabelEncoders
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train/test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "customer_churn_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model and encoders saved successfully!")
