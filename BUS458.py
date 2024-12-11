import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("decision_tree_model4.pkl", "rb") as file:
    DTmodel = pickle.load(file)

# Title for the app
st.title("Data Scientist Salary Prediction")

# Inputs
st.header("Enter Your Information")
country = st.selectbox("Do you live in the USA? (COUNTRY)", ["Yes", "No"])
role = st.selectbox("Do you work in the technology/online services industry? (ROLE)", ["Yes", "No"])
ml = st.selectbox("Does your current employer incorporate machine learning methods into their business? (ML)", ["Yes", "No"])
proto = st.selectbox("Do you use machine learning prototyping services? (PROTO)", ["Yes", "No"])
years = st.selectbox("Do you have 20 or more years of experience in coding? (YEARS)", ["Yes", "No"])

# Convert "Yes" and "No" inputs to 1 and 0
country = 1 if country == "Yes" else 0
role = 1 if role == "Yes" else 0
ml = 1 if ml == "Yes" else 0
proto = 1 if proto == "Yes" else 0
years = 1 if years == "Yes" else 0

# Dynamically match input data to model's expected features
top_features = [
    'Country_Category_United States of America',
    'industry_category_Technology/Online Services',
    'ml_methods_use_Uncertain/No ML Methods',
    'ML_Prototyping_Services',
    'Years Coding_20+ years'
]

input_data_encoded = pd.DataFrame(columns=top_features)
input_data_encoded.loc[0] = 0  # Initialize all features to 0

# Assign user inputs
input_data_encoded['Country_Category_United States of America'] = country
input_data_encoded['industry_category_Technology/Online Services'] = role
input_data_encoded['ml_methods_use_Uncertain/No ML Methods'] = ml
input_data_encoded['ML_Prototyping_Services'] = proto
input_data_encoded['Years Coding_20+ years'] = years

# Predict
if st.button("What is my Data Scientist Salary?"):
    try:
        prediction = DTmodel.predict(input_data_encoded)[0]
        st.success(f"Your predicted salary as a Data Scientist is: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")



  

