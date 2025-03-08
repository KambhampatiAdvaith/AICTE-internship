import streamlit as st
import pickle
import numpy as np

# Load trained models
model_files = {
    'diabetes': 'diabetes_model.pkl',
    'heart_disease': 'heart_disease_model.pkl',
    'parkinson': 'parkinson_model.pkl',
    'lung_cancer': 'lung_cancer_model.pkl'
}

models = {}
for key, file in model_files.items():
    with open(file, 'rb') as f:
        models[key] = pickle.load(f)

st.title('AI-Powered Medical Diagnosis System')

# Sidebar for navigation
st.sidebar.title("Disease Prediction")
option = st.sidebar.radio("Select Disease to Predict", ["Diabetes", "Heart Disease", "Parkinson's Disease", "Lung Cancer"])


if option == 'Diabetes':
    st.header('Diabetes Prediction')
    pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
    glucose = st.number_input('Glucose Level', min_value=0)
    blood_pressure = st.number_input('Blood Pressure', min_value=0)
    skin_thickness = st.number_input('Skin Thickness', min_value=0)
    insulin = st.number_input('Insulin Level', min_value=0)
    bmi = st.number_input('BMI', min_value=0.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    age = st.number_input('Age', min_value=0, step=1)
    
    if st.button('Diabetes Test Result'):
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
        prediction = models['diabetes'].predict(input_data)[0]
        result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
        st.success(f'Result: {result}')

elif option == 'Heart Disease':
    st.header('Heart Disease Prediction')
    age = st.number_input('Age', min_value=0, step=1)
    sex = st.number_input('Sex (1=Male, 0=Female)', min_value=0, max_value=1)
    cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3)
    trestbps = st.number_input('Resting Blood Pressure', min_value=0)
    chol = st.number_input('Cholesterol', min_value=0)
    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', min_value=0, max_value=1)
    restecg = st.number_input('Resting ECG (0-2)', min_value=0, max_value=2)
    thalach = st.number_input('Max Heart Rate Achieved', min_value=0)
    exang = st.number_input('Exercise Induced Angina (1=Yes, 0=No)', min_value=0, max_value=1)
    oldpeak = st.number_input('ST Depression Induced', min_value=0.0)
    slope = st.number_input('Slope of Peak Exercise ST Segment (0-2)', min_value=0, max_value=2)
    ca = st.number_input('Major Vessels (0-4)', min_value=0, max_value=4)
    thal = st.number_input('Thalassemia (1-3)', min_value=1, max_value=3)
    
    if st.button('Heart Disease Test Result'):
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = models['heart_disease'].predict(input_data)[0]
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
        st.success(f'Result: {result}')

elif option == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")
    fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0)
    fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0)
    flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0)
    jitter = st.number_input('Jitter(%)', min_value=0.0)
    shimmer = st.number_input('Shimmer', min_value=0.0)
    hnr = st.number_input('HNR', min_value=0.0)
    rpde = st.number_input('RPDE', min_value=0.0)
    dfa = st.number_input('DFA', min_value=0.0)
    spread1 = st.number_input('Spread1', min_value=-10.0)
    spread2 = st.number_input('Spread2', min_value=0.0)
    d2 = st.number_input('D2', min_value=0.0)
    ppe = st.number_input('PPE', min_value=0.0)
    
    if st.button("Parkinson's Test Result"):
        input_data = np.array([fo, fhi, flo, jitter, shimmer, hnr, rpde, dfa, spread1, spread2, d2, ppe]).reshape(1, -1)
        prediction = models['parkinson'].predict(input_data)[0]
        result = "Parkinson's Disease Detected" if prediction == 1 else "No Parkinson's Disease"
        st.success(f'Result: {result}')

elif option == 'Lung Cancer':
    st.header("Lung Cancer Prediction")
    age = st.number_input('Age', min_value=0, step=1)
    gender = st.number_input('Gender (1=Male, 0=Female)', min_value=0, max_value=1)
    smoking = st.number_input('Smoking (1=Yes, 0=No)', min_value=0, max_value=1)
    yellow_fingers = st.number_input('Yellow Fingers (1=Yes, 0=No)', min_value=0, max_value=1)
    anxiety = st.number_input('Anxiety (1=Yes, 0=No)', min_value=0, max_value=1)
    chronic_disease = st.number_input('Chronic Disease (1=Yes, 0=No)', min_value=0, max_value=1)
    fatigue = st.number_input('Fatigue (1=Yes, 0=No)', min_value=0, max_value=1)
    allergy = st.number_input('Allergy (1=Yes, 0=No)', min_value=0, max_value=1)
    wheezing = st.number_input('Wheezing (1=Yes, 0=No)', min_value=0, max_value=1)
    coughing = st.number_input('Coughing (1=Yes, 0=No)', min_value=0, max_value=1)
    
    if st.button("Lung Cancer Test Result"):
        input_data = np.array([age, gender, smoking, yellow_fingers, anxiety, chronic_disease, fatigue, allergy, wheezing, coughing]).reshape(1, -1)
        prediction = models['lung_cancer'].predict(input_data)[0]
        result = "Lung Cancer Detected" if prediction == 1 else "No Lung Cancer"
        st.success(f'Result: {result}')
