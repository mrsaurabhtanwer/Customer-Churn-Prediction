import streamlit as st
import pandas as pd
import numpy as np
import pickle 

# Load the trained model
with open('churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

# Collect user input features via Streamlit
st.sidebar.header('Enter Customer Information')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.sidebar.selectbox('Partner', ['No', 'Yes'])
    dependents = st.sidebar.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 24)
    monthly_charges = st.sidebar.number_input('Monthly Charges', 0.0, 150.0, 50.0)
    total_charges = st.sidebar.number_input('Total Charges', 0.0, 10000.0, 1000.0)
    # Add more input features as required
    data = {'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode user input values (matching encoding used during training)
input_df['gender'] = input_df['gender'].map({'Female': 0, 'Male': 1})
input_df['SeniorCitizen'] = input_df['SeniorCitizen'].map({'No': 0, 'Yes': 1})
input_df['Partner'] = input_df['Partner'].map({'No': 0, 'Yes': 1})
input_df['Dependents'] = input_df['Dependents'].map({'No': 0, 'Yes': 1})

# Make prediction using the trained model
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display the prediction
st.subheader('Prediction:')
if prediction == 1:
    st.write("🔴 Customer is likely to churn")
else:
    st.write("🟢 Customer is unlikely to churn")

# Show prediction probability
st.subheader('Prediction Probability:')
st.write(f"Churn Probability: {prediction_proba[0][1]:.2f}")
st.write(f"Non-Churn Probability: {prediction_proba[0][0]:.2f}")

# Provide retention strategy
if prediction == 1:
    st.subheader('Recommended Retention Strategy:')
    st.write("Offer personalized discounts or contact customer support to resolve issues.")
else:
    st.subheader('Customer Retention Strategy:')
    st.write("Continue providing excellent service and periodic offers.")
