import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Configuration
# -------------------------

MODEL_DIR = 'models'
USER_RECORD_DIR = 'user_records'
CREDENTIALS_FILE = 'credentials.csv'
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['AWD', 'D', 'NED'])  # Set your target classes manually

# -------------------------
# Authentication Functions
# -------------------------

def load_credentials():
    if os.path.exists(CREDENTIALS_FILE):
        return pd.read_csv(CREDENTIALS_FILE)
    else:
        return pd.DataFrame(columns=['username', 'password'])

def check_login(username, password):
    creds = load_credentials()
    user = creds[(creds['username'] == username) & (creds['password'] == password)]
    return not user.empty

def save_record(username, input_data, prediction):
    user_folder = os.path.join(USER_RECORD_DIR, username)
    os.makedirs(user_folder, exist_ok=True)
    record_file = os.path.join(user_folder, 'record.csv')

    df = pd.DataFrame([input_data])
    df['Prediction'] = prediction
    df['Timestamp'] = datetime.now()

    if os.path.exists(record_file):
        existing = pd.read_csv(record_file)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(record_file, index=False)

# -------------------------
# Model Loading
# -------------------------

@st.cache_resource
def load_model_file(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    return load_model(model_path)

# -------------------------
# UI Components
# -------------------------

def login_page():
    st.title("🔒 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
            time.sleep(1)
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

def main_page():
    st.title(f"Welcome, {st.session_state.username} 👋")
    st.subheader("Enter Patient Details")

    # Form inputs
    sex = st.selectbox('Sex', ['Male', 'Female'])
    grade = st.selectbox('Grade', ['Intermediate', 'High'])
    hist_type = st.selectbox('Histological Type', [
        'pleiomorphic leiomyosarcoma', 'malignant solitary fibrous tumor',
        'sclerosing epithelioid fibrosarcoma', 'myxoid fibrosarcoma',
        'undifferentiated - pleiomorphic', 'synovial sarcoma',
        'undifferentiated pleomorphic liposarcoma', 'epithelioid sarcoma',
        'poorly differentiated synovial sarcoma',
        'pleiomorphic spindle cell undifferentiated',
        'pleomorphic sarcoma', 'myxofibrosarcoma', 'leiomyosarcoma'
    ])
    mskcc_type = st.selectbox('MSKCC Type', ['MFH', 'Synovial sarcoma', 'Leiomyosarcoma'])
    site = st.selectbox('Site of Primary STS', [
        'left thigh', 'right thigh', 'right parascapusular', 'left biceps',
        'right buttock', 'parascapusular', 'left buttock'
    ])
    treatment = st.selectbox('Treatment', [
        'Radiotherapy + Surgery', 
        'Radiotherapy + Surgery + Chemotherapy', 
        'Surgery + Chemotherapy'
    ])

    model_choice = st.selectbox("Select Model", ['Simple ANN', 'Deep ANN', 'CNN', 'LSTM'])

    if st.button("Predict"):
        input_data = {
            'Sex': sex,
            'Grade': grade,
            'Histological type': hist_type,
            'MSKCC type': mskcc_type,
            'Site of primary STS': site,
            'Treatment': treatment
        }

        # Prepare input for model
        input_df = pd.DataFrame([input_data])

        # One-hot encoding
        input_encoded = pd.get_dummies(input_df)
        full_columns = joblib.load('models/input_columns.pkl')  # Pre-saved during training
        input_encoded = input_encoded.reindex(columns=full_columns, fill_value=0)

        X_input = input_encoded.values.astype(np.float32)

        model_filename = model_choice.replace(" ", "_").lower() + '.h5'
        model = load_model_file(model_filename)

        preds = model.predict(X_input)
        predicted_class = label_encoder.inverse_transform(np.argmax(preds, axis=1))[0]

        st.success(f"Predicted Status: **{predicted_class}**")

        # Save the record
        save_record(st.session_state.username, input_data, predicted_class)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()

# -------------------------
# Streamlit Flow Control
# -------------------------

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if st.session_state.logged_in:
    main_page()
else:
    login_page()
