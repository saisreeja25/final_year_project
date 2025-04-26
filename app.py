import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, GlobalMaxPooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import hashlib

# --- Helper Functions for User Authentication ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(stored_password, input_password):
    return stored_password == hash_password(input_password)

# --- Helper Functions for User Registration ---
users_db = {
    "admin": hash_password("adminpassword")  # Example stored user (admin)
}

def register_user(username, password):
    if username in users_db:
        return False  # User already exists
    users_db[username] = hash_password(password)
    return True

# --- Helper Function for Training ---
def load_label_encoder():
    return joblib.load('label_encoder.pkl')

def load_feature_columns():
    return joblib.load('input_columns.pkl')

def load_models():
    models = {
        'Simple ANN': load_model('Simple ANN_model.h5'),
        'CNN': load_model('CNN1_model.h5'),
        'LSTM': load_model('LSTM1_model.h5')
    }
    return models

def build_simple_ann(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # For three classes
    ])
    return model

def build_cnn(input_shape):
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        Conv1D(64, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # For three classes
    ])
    return model

def build_lstm(input_shape):
    model = Sequential([
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # For three classes
    ])
    return model

# --- Main Function ---
def main_page():
    st.title('Medical Prediction App')

    # User input for login or register
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        option = st.radio("Choose an option", ["Login", "Register"])
        if option == "Login":
            login_page()
        else:
            register_page()
    else:
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button('Logout'):
            st.session_state.logged_in = False
            del st.session_state.username
            st.success("Logged out successfully!")
            #st.experimental_rerun()  # Reload the page

        st.header('Main Content')

        # Tab selection for training and prediction
        selected_tab = st.radio('Choose an option', ['Train Model', 'Make Prediction'])

        if selected_tab == 'Train Model':
            train_page()
        elif selected_tab == 'Make Prediction':
            prediction_page()

# --- Register Page Function ---
def register_page():
    st.title('Register Page')

    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    confirm_password = st.text_input('Confirm Password', type='password')

    if password != confirm_password:
        st.error("Passwords do not match!")
    elif st.button('Register'):
        if register_user(username, password):
            st.success(f"User {username} registered successfully!")
            st.session_state.logged_in = True
            st.session_state.username = username
            #st.experimental_rerun()  # Reload the page
        else:
            st.error(f"User {username} already exists!")

# --- Login Page Function ---
def login_page():
    st.title('Login Page')

    # User input for login
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    # Login button
    if st.button('Login'):
        if username in users_db and check_password(users_db[username], password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome back, {username}!")
            st.experimental_rerun()  # Reload the page
        else:
            st.error('Invalid credentials!')

# --- Train Model Page ---
def train_page():
    st.subheader('Train a Model')

    # Upload dataset
    file = st.file_uploader("Upload CSV dataset", type="csv")
    
    if file is not None:
        df = pd.read_csv(file)
        st.write("Dataset preview:")
        st.write(df.head())

        # Drop 'Patient ID' column if it exists
        if 'Patient ID' in df.columns:
            df.drop('Patient ID', axis=1, inplace=True)

        # Preprocess dataset
        X = df.drop(columns=['Status (NED, AWD, D)'])  # Assuming 'Status' is the target column
        y = df['Status (NED, AWD, D)']

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler for later use
        joblib.dump(scaler, 'scaler.pkl')

        # Convert target variable to one-hot encoding for multi-class classification
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)

        # Select model type based on user input
        model_choice = st.selectbox('Select a Model Type', ['Simple ANN', 'CNN', 'LSTM'])

        # Define model based on selected choice
        if model_choice == 'Simple ANN':
            model = build_simple_ann(X_train.shape[1])
        elif model_choice == 'CNN':
            model = build_cnn(X_train.shape[1])
        elif model_choice == 'LSTM':
            model = build_lstm(X_train.shape[1])

        # Compile the model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Save the trained model
        model.save(f'{model_choice}_model.h5')
        st.success(f"Model '{model_choice}' has been trained and saved successfully!")

        # Display training history
        st.subheader('Training History')
        st.line_chart(pd.DataFrame(history.history)[['accuracy', 'val_accuracy']])

        # Show model summary
        st.subheader('Model Summary')
        model.summary()

        # Option to download the trained model
        st.download_button(
            label="Download Trained Model",
            data=open(f'{model_choice}_model.h5', 'rb').read(),
            file_name=f'{model_choice}_model.h5',
            mime="application/octet-stream"
        )

# --- Prediction Page ---def prediction_page():
    st.subheader('Make Prediction')

    # Load models
    models = load_models()

    # Load other components like columns, label encoder, etc.
    full_columns = load_feature_columns()  # Define or load the feature columns
    label_encoder = load_label_encoder()  # Load label encoder
    scaler = joblib.load('scaler.pkl')  # Load the scaler for Age column scaling

    # User input for prediction
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120)
    grade = st.selectbox('Grade', ['Intermediate', 'High'])
    histological_type = st.selectbox('Histological Type', ['pleiomorphic leiomyosarcoma', 'malignant solitary fibrous tumor', 'sclerosing epithelioid fibrosarcoma', 'myxoid fibrosarcoma', 'undifferentiated - pleiomorphic', 'synovial sarcoma', 'undifferentiated pleomorphic liposarcoma', 'epithelioid sarcoma', 'poorly differentiated synovial sarcoma', 'pleiomorphic spindle cell undifferentiated', 'pleomorphic sarcoma', 'myxofibrosarcoma', 'leiomyosarcoma'])
    mskcc_type = st.selectbox('MSKCC Type', ['MFH', 'Synovial sarcoma', 'Leiomyosarcoma'])
    site_of_primary = st.selectbox('Site of Primary STS', ['left thigh', 'right thigh', 'right parascapusular', 'left biceps', 'right buttock', 'parascapusular', 'left buttock'])
    treatment = st.selectbox('Treatment', ['Radiotherapy + Surgery', 'Radiotherapy + Surgery + Chemotherapy', 'Surgery + Chemotherapy'])

    input_data = np.array([sex, age, grade, histological_type, mskcc_type, site_of_primary, treatment])

    # Preprocess the input data
    input_data = input_data.reshape(1, -1)  # Reshape it to 2D for the scaler

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make prediction using the selected model
    model_choice = st.selectbox('Select a Model', list(models.keys()))
    model = models[model_choice]

    # Make prediction and display result
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)

    st.write(f"Predicted Class: {label_encoder.inverse_transform(predicted_class)}")


if __name__ == "__main__":
    main_page()
