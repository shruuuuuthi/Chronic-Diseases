import streamlit as st
import pandas as pd
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
label_encoders = data["label_encoders"]

def show_predict_page():
    st.title("Chronic Disease Prediction")

    st.write("""### We need some information to predict the chronic disease""")

    columns = ['Gender', 'Family History of Chronic Disease', 'Diet', 'Physical Activity',
               'Alcohol (units/week)', 'Smoking (cigarettes/day)', 'Stress Level (1-10)',
               'Blood Pressure (mmHg)', 'Sleep (hours/night)', 'Cholesterol (mg/dL)',
               'Past Medical Condition', 'BMI', 'Ethnicity/Race', 'Blood Sugar (mg/dL)',
               'Heart Rate (bpm)', 'Type of Work', 'Education Level', 'Work Status',
               'Marital Status', 'Place of Residence']

    options = {}
    for col in columns:
        options[col] = st.selectbox(col, data[col])

    ok = st.button("Predict Disease")
    if ok:
        X = []
        for col in columns:
            X.append(options[col])

        X = np.array([X])
        for i, col in enumerate(columns):
            X[:, i] = label_encoders[col].transform(X[:, i])

        X = X.astype(float)
        predictions = regressor.predict(X)
        st.subheader(f"The predicted disease is: {predictions[0]}")

# Render the predict page
show_predict_page()
