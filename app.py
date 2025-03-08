import streamlit as st
import pickle
import numpy as np

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load models
diabetes_model = load_model('diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model('parkinsons_model.sav')

def predict_diabetes(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = diabetes_model.predict(input_array)
    return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

def predict_heart_disease(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = heart_disease_model.predict(input_array)
    return 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'

def predict_parkinsons(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = parkinsons_model.predict(input_array)
    return 'Parkinsons Disease Detected' if prediction[0] == 1 else 'No Parkinsons Disease'

st.title("Disease Prediction System")

option = st.selectbox("Select a Disease to Predict", ["Diabetes", "Heart Disease", "Parkinson's"])

if option == "Diabetes":
    st.header("Diabetes Prediction")
    Pregnancies = st.number_input("Pregnancies", min_value=0)
    Glucose = st.number_input("Glucose Level", min_value=0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0)
    Insulin = st.number_input("Insulin", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    Age = st.number_input("Age", min_value=0)
    
    if st.button("Predict Diabetes"):
        result = predict_diabetes([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(result)

elif option == "Heart Disease":
    st.header("Heart Disease Prediction")
    Age = st.number_input("Age", min_value=0)
    Sex = st.number_input("Sex (1=Male, 0=Female)", min_value=0, max_value=1)
    ChestPainType = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
    RestingBP = st.number_input("Resting Blood Pressure", min_value=0)
    Cholesterol = st.number_input("Cholesterol", min_value=0)
    FastingBS = st.number_input("Fasting Blood Sugar (1=True, 0=False)", min_value=0, max_value=1)
    RestingECG = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2)
    MaxHR = st.number_input("Max Heart Rate", min_value=0)
    ExerciseAngina = st.number_input("Exercise-Induced Angina (1=True, 0=False)", min_value=0, max_value=1)
    Oldpeak = st.number_input("ST Depression", min_value=0.0)
    ST_Slope = st.number_input("ST Slope (0-2)", min_value=0, max_value=2)
    
    if st.button("Predict Heart Disease"):
        result = predict_heart_disease([Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope])
        st.success(result)

elif option == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    Fo = st.number_input("Fo (Fundamental Frequency)", min_value=0.0)
    Fhi = st.number_input("Fhi (Highest Frequency)", min_value=0.0)
    Flo = st.number_input("Flo (Lowest Frequency)", min_value=0.0)
    Jitter_percent = st.number_input("Jitter (%)", min_value=0.0)
    Jitter_Abs = st.number_input("Jitter (Abs)", min_value=0.0)
    RAP = st.number_input("RAP", min_value=0.0)
    PPQ = st.number_input("PPQ", min_value=0.0)
    Shimmer = st.number_input("Shimmer", min_value=0.0)
    Shimmer_dB = st.number_input("Shimmer (dB)", min_value=0.0)
    HNR = st.number_input("HNR", min_value=0.0)
    NHR = st.number_input("NHR", min_value=0.0)
    RPDE = st.number_input("RPDE", min_value=0.0)
    DFA = st.number_input("DFA", min_value=0.0)
    Spread1 = st.number_input("Spread1", min_value=0.0)
    Spread2 = st.number_input("Spread2", min_value=0.0)
    PPE = st.number_input("PPE", min_value=0.0)
    
    if st.button("Predict Parkinson's"):
        result = predict_parkinsons([Fo, Fhi, Flo, Jitter_percent, Jitter_Abs, RAP, PPQ, Shimmer, Shimmer_dB, HNR, NHR, RPDE, DFA, Spread1, Spread2, PPE])
        st.success(result)
