import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Bank Churn Prediction",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")

st.write("Enter customer details")

credit_score = st.number_input(
    "Credit Score",
    300,
    900,
    600
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    18,
    100,
    35
)

tenure = st.number_input(
    "Tenure",
    0,
    10,
    5
)

balance = st.number_input(
    "Balance",
    0.0
)

num_products = st.number_input(
    "Number Of Products",
    1,
    4,
    1
)

has_card = st.selectbox(
    "Has Credit Card",
    [0,1]
)

active_member = st.selectbox(
    "Is Active Member",
    [0,1]
)

salary = st.number_input(
    "Estimated Salary",
    0.0
)

if st.button("Predict"):

    try:

        ratio = balance / (salary + 1)

        ppt = num_products / (tenure + 1)

        if age <= 30:
            age_group = "Young"
        elif age <= 45:
            age_group = "Adult"
        elif age <= 60:
            age_group = "MiddleAge"
        else:
            age_group = "Senior"

        data = pd.DataFrame({
            'CreditScore':[credit_score],
            'Geography':[geography],
            'Gender':[gender],
            'Age':[age],
            'Tenure':[tenure],
            'Balance':[balance],
            'NumOfProducts':[num_products],
            'HasCrCard':[has_card],
            'IsActiveMember':[active_member],
            'EstimatedSalary':[salary],
            'BalanceSalaryRatio':[ratio],
            'ProductsPerTenure':[ppt],
            'AgeGroup':[age_group]
        })

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error(
                "Customer Will Leave Bank"
            )
        else:
            st.success(
                "Customer Will Stay"
            )

    except Exception as e:
        st.error(str(e))