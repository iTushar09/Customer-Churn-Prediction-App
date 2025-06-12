# app1.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from churn_model import ChurnModel

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    churn_model = ChurnModel()
    try:
        churn_model.load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return churn_model

model = load_model()

st.title("📊 Customer Churn Prediction Dashboard")
tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Importance", "About"])

with tab1:
    st.header("Customer Details")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 50.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with col2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges)
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    with st.expander("Additional Features"):
        col3, col4 = st.columns(2)
        with col3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with col4:
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "InternetService": internet_service,
        "OnlineBackup": online_backup,
        "PaperlessBilling": paperless_billing
    }
if st.button("Predict Churn"):
    if model is None:
        st.error("Model not available.")
    else:
        try:
            prediction, proba = model.predict(input_data)
            churn_status = "Churn Risk" if prediction == 1 else "Likely to Stay"
            confidence = proba[1] if prediction == 1 else proba[0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"### {churn_status}")
            else:
                st.success(f"### {churn_status}")
            st.write(f"Confidence: {confidence * 100:.2f}%")

            # Explanation
            with st.expander("🔍 What does this mean?"):
                if prediction == 1:
                    st.markdown("""
                    - **Churn Risk** means the customer is **likely to leave** the service.
                    - **Confidence** reflects how sure the model is about this prediction.
                    - For example, a confidence of 80% means the model is 80% sure the customer will churn.
                    """)
                else:
                    st.markdown("""
                    - **Likely to Stay** means the customer is **predicted to continue** using the service.
                    - **Confidence** shows how certain the model is about this.
                    - A confidence of 57% means there's still a **43% chance they might churn**, so it's worth monitoring.
                    """)

            # Confidence bar chart
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.barh([""], [confidence * 100], color='red' if prediction == 1 else 'green')
            ax.barh([""], [100 - confidence * 100], left=[confidence * 100], color='lightgray')
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel('Confidence %')
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


with tab2:
    st.header("Feature Importance")
    if model and model.model and hasattr(model.model, 'feature_importances_'):
        importances = model.model.feature_importances_
        features = model.model.feature_names_in_ if hasattr(model.model, 'feature_names_in_') else model.feature_names
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(imp_df['Feature'], imp_df['Importance'], color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.dataframe(imp_df)
    else:
        st.warning("Feature importance not available.")

with tab3:
    st.header("About This App")
    st.write("""
    This tool predicts customer churn using a machine learning model trained from a Jupyter Notebook.

    **Features:**
    - Accepts user inputs and predicts churn risk.
    - Displays confidence and feature importance.
    - Simple, interactive UI for ease of use.

    **Technologies Used:**
     - Python, Streamlit, NumPy, pandas, matplotlib, seaborn, scikit-learn, SMOTE (imbalanced-learn), XGBoost, pickle.
   
    **Created by:** Tushar Chaudhari
    """)
    st.write("For more information, visit the [GitHub repository](https://github.com/iTushar09/Customer-Churn-Prediction-App.git)")