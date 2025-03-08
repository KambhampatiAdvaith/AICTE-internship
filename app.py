import streamlit as st
import pickle
import numpy as np
import os

# Load Models Function
def load_model(model_name):
    try:
        with open(model_name, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.warning(f"Error loading {model_name}: {e}")
        return None

# Load Models
models = {
    "diabetes": load_model("diabetes_model.pkl"),
    "heart_disease": load_model("heart_disease_model.pkl"),
    "parkinsons": load_model("parkinsons_model.pkl"),
    "lung_cancer": load_model("lung_cancer_model.pkl"),
    "thyroid": load_model("thyroid_model.pkl")
}

# Styling
st.markdown("""
    <style>
        .main { background: url('https://source.unsplash.com/1600x900/?medical,health') no-repeat center center fixed; 
                background-size: cover; }
        .stApp { background-color: rgba(0, 0, 0, 0.4); padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("AI-Powered Disease Prediction System")
st.markdown("Enter your details below to get a preliminary disease prediction.")

# Disease Selection
disease_option = st.selectbox("Select a disease to check:", list(models.keys()))

# Input Fields
if disease_option == "diabetes":
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
    bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=80)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=300, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    features = np.array([[age, glucose, bp, insulin, bmi]])

elif disease_option == "heart_disease":
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=140)
    features = np.array([[age, cholesterol, bp, max_hr]])

elif disease_option == "parkinsons":
    tremor = st.selectbox("Do you have tremors?", ["Yes", "No"])  
    speech = st.selectbox("Do you have speech issues?", ["Yes", "No"]) 
    features = np.array([[1 if tremor == "Yes" else 0, 1 if speech == "Yes" else 0]])

elif disease_option == "lung_cancer":
    smoking = st.selectbox("Do you smoke?", ["Yes", "No"])  
    cough = st.selectbox("Do you have a persistent cough?", ["Yes", "No"]) 
    features = np.array([[1 if smoking == "Yes" else 0, 1 if cough == "Yes" else 0]])

elif disease_option == "thyroid":
    tsh = st.number_input("TSH Level", min_value=0.1, max_value=10.0, value=2.5)
    t3 = st.number_input("T3 Level", min_value=0.1, max_value=5.0, value=1.2)
    features = np.array([[tsh, t3]])

# Predict Button
if st.button("Predict"):
    model = models[disease_option]
    if model:
        prediction = model.predict(features)
        result = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"Prediction Result: {result}")
        
        if result == "Positive":
            st.warning("This is an AI-based prediction. Consult a doctor for further evaluation.")
    else:
        st.error("Model not available. Please try again later.")
