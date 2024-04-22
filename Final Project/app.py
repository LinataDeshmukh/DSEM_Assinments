import streamlit as st
import pandas as pd
import joblib

# Function to load a saved model
def load_model(model_name):
    # Adjust the path if your models are saved in a different directory
    return joblib.load(f'{model_name.replace(" ", "_")}_model.pkl')

# Title of the application
st.title('Heart Disease Prediction Application')

# Description
st.write("This application predicts the likelihood of a heart disease based on input parameters. \
         It uses the top 3 performing models: Gradient Boosting, Random Forest, and SVM.")

# Model names list (make sure these names match the names used in your training script)
model_names = ["Gradient Boosting", "Random Forest", "SVM"]

# Load models (you might need to adjust this if your environment differs)
models = {name: load_model(name) for name in model_names}

# User input fields
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 50)
    sex = st.sidebar.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.sidebar.slider('Resting BP', 90, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol', 100, 600, 200)
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
    resting_ecg = st.sidebar.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.sidebar.slider('Maximum Heart Rate', 60, 220, 150)
    exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', ('Y', 'N'))
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 2.0)
    st_slope = st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))

    # Create a DataFrame of the input features
    data = {
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# Display the user input features
st.subheader('User Input parameters for prediction')
st.write(input_df)

# Button to perform prediction
if st.button('Predict Heart Disease'):
    st.subheader('Prediction results')
    for name, model in models.items():
        prediction = model.predict(input_df)
        st.write(f"{name}: {'Heart Disease' if prediction[0] == 1 else 'Normal'}")
