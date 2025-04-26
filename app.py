import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Load the models and scalers
models = {
    'Simple ANN': joblib.load('Simple ANN_model.pkl'),
    'Deep ANN': joblib.load('Deep ANN_model.pkl'),
    'CNN': joblib.load('CNN_model.pkl'),
    'LSTM': joblib.load('LSTM_model.pkl')
}

scaler = joblib.load('scaler.pkl')  # Assuming the scaler is saved in this file
label_encoder = joblib.load('label_encoder.pkl')  # Assuming the label encoder is saved here

# User authentication
def login():
    st.title("Login Page")
    
    # Form for login
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            user_df = pd.read_csv("user_records.csv")  # User data
            if username in user_df['Username'].values:
                user_data = user_df[user_df['Username'] == username]
                if user_data.iloc[0]['Password'] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    return True
                else:
                    st.error("Incorrect password.")
            else:
                st.error("Username not found.")
    return False

# Registration page
def register():
    st.title("Registration Page")
    
    with st.form(key="register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")
        
        if submit_button:
            if password == confirm_password:
                user_df = pd.read_csv("user_records.csv")
                new_user = pd.DataFrame({"Username": [username], "Password": [password]})
                user_df = pd.concat([user_df, new_user], ignore_index=True)
                user_df.to_csv("user_records.csv", index=False)
                st.success("Registration successful! You can now login.")
            else:
                st.error("Passwords do not match.")

# Preprocess user input
def preprocess_user_input(user_input):
    categorical_cols = ['Sex', 'Grade', 'Histological type', 'MSKCC type', 'Site of primary STS', 'Treatment']
    input_df = pd.DataFrame([user_input])
    
    # Handle numerical columns
    input_df['Age'] = scaler.transform(input_df[['Age']])
    
    # One hot encoding for categorical columns
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    return input_df.astype(np.float32).values

# Predict function for model
def predict(model_name, user_input):
    model = models[model_name]
    X_input = preprocess_user_input(user_input)
    prediction = model.predict(X_input)
    decoded_prediction = label_encoder.inverse_transform(prediction)
    return decoded_prediction[0]

# Main page
def main_page():
    st.title("Model Prediction Page")

    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.error("Please log in first.")
        return

    st.write(f"Welcome {st.session_state.username}!")

    # Input fields for prediction
    user_input = {
        'Sex': st.selectbox('Sex', ['Male', 'Female']),
        'Age': st.number_input('Age', min_value=0, max_value=100),
        'Grade': st.selectbox('Grade', ['High', 'Intermediate']),
        'Histological type': st.selectbox('Histological type', ['leiomyosarcoma', 'synovial sarcoma', 'fibrosarcoma']),
        'MSKCC type': st.selectbox('MSKCC type', ['MFH', 'Synovial sarcoma', 'Leiomyosarcoma']),
        'Site of primary STS': st.selectbox('Site of primary STS', ['left thigh', 'right thigh', 'left biceps', 'right buttock']),
        'Treatment': st.selectbox('Treatment', ['Radiotherapy + Surgery', 'Radiotherapy + Surgery + Chemotherapy', 'Surgery + Chemotherapy'])
    }

    model_choice = st.selectbox("Choose model", ['Simple ANN', 'Deep ANN', 'CNN', 'LSTM'])
    
    if st.button("Predict"):
        prediction = predict(model_choice, user_input)
        st.write(f"Prediction: {prediction}")

# Logout page
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("You have logged out successfully.")

# Main app flow
def app():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Select an option", ["Login", "Register", "Predict", "Logout"])
    
    if choice == "Login":
        if login():
            main_page()
    elif choice == "Register":
        register()
    elif choice == "Predict":
        if "logged_in" in st.session_state and st.session_state.logged_in:
            main_page()
        else:
            st.error("Please log in first.")
    elif choice == "Logout":
        logout()

if __name__ == "__main__":
    app()
