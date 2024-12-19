import streamlit as st
import joblib
import numpy as np

# Load the trained machine learning model
# Ensure "diabetes_model.pkl" is in the same directory as this file
model = joblib.load("diabetes_model.pkl")

# Title of the Streamlit App
st.title("Diabetes Prediction App")

# User Input Fields for the Features
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, format="%d")
glucose = st.number_input("Glucose Level", min_value=0.0, format="%.2f")
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, format="%.2f")
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, format="%.2f")
insulin = st.number_input("Insulin Level", min_value=0.0, format="%.2f")
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, step=1, format="%d")

# Button to Trigger Prediction
if st.button("Predict"):
    # Prepare input data as a 2D array for the model
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display Prediction Result
    if prediction[0] == 1:
        st.write("### 🎯 The model predicts: **Diabetic**")
    else:
        st.write("### ✅ The model predicts: **Non-Diabetic**")

# Footer
st.write("Built with ❤️ using Streamlit")
