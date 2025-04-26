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

# Function to load pre-trained models

import joblib

# Function to load the label encoder
def load_label_encoder():
    return joblib.load('label_encoder.pkl')
# Function to load the feature columns
def load_feature_columns():
    return joblib.load('input_columns.pkl')  # Load the feature columns from a pickle file

def load_models():
    models = {
        'Simple ANN': load_model('Simple ANN_model.h5'),
        'CNN': load_model('CNN1_model.h5'),
        'LSTM': load_model('LSTM1_model.h5')
    }
    return models
# --- Model Building Functions ---
def build_simple_ann(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # For three classes
    ])
    return model

def build_deep_ann(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
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

# --- Train Page Function ---
def train_page():
    st.title('Train Model')

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
        model_choice = st.selectbox('Select a Model Type', ['Simple ANN', 'Deep ANN', 'CNN', 'LSTM'])

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

# --- Main Page Function ---
def main_page():
    st.title('Medical Prediction App')
    
    # Load models
    models = load_models()  # Load pre-trained models here

    # Load other components like columns, label encoder, etc.
    full_columns = load_feature_columns()  # Define or load the feature columns
    label_encoder = load_label_encoder()  # Load label encoder
    scaler = joblib.load('scaler.pkl')  # Load the scaler for Age column scaling

    # Sidebar for login or train model
    st.sidebar.title('Login / Register')
    page = st.sidebar.radio('Choose Page:', ['Login', 'Register', 'Train Model'])

    if page == 'Login':
        st.sidebar.header('Login')
        username = st.sidebar.text_input('Username')
        password = st.sidebar.text_input('Password', type='password')
        
        if st.sidebar.button('Login'):
            if check_login(username, password):
                st.success('Login Successful!')
                user_input_form(models, full_columns, label_encoder, scaler, username)
            else:
                st.error('Invalid username or password')

    elif page == 'Register':
        st.sidebar.header('Register')
        username = st.sidebar.text_input('New Username')
        password = st.sidebar.text_input('New Password', type='password')

        if st.sidebar.button('Register'):
            if save_new_user(username, password):
                st.success(f'Account created for {username}')
            else:
                st.error('Username already exists')

    elif page == 'Train Model':
        train_page()  # Navigate to train model page
# --- Helper Functions for User Authentication ---
def check_login(username, password):
    if os.path.exists('credentials.csv'):
        credentials = pd.read_csv('credentials.csv')
        user_data = credentials[credentials['username'] == username]
        if not user_data.empty and user_data['password'].values[0] == password:
            return True
    return False

def save_new_user(username, password):
    if not os.path.exists('credentials.csv'):
        df = pd.DataFrame(columns=['username', 'password'])
        df.to_csv('credentials.csv', index=False)

    credentials = pd.read_csv('credentials.csv')

    if username in credentials['username'].values:
        return False  # Username already exists
    else:
        new_user = pd.DataFrame({'username': [username], 'password': [password]})
        credentials = pd.concat([credentials, new_user], ignore_index=True)
        credentials.to_csv('credentials.csv', index=False)
        return True

# --- Helper Function for Prediction ---
def user_input_form(models, full_columns, label_encoder, scaler, username):
    st.header('Enter Patient Information for Prediction')

    # Input fields for user
    sex = st.selectbox('Sex', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120)
    grade = st.selectbox('Grade', ['Intermediate', 'High'])
    histological_type = st.selectbox('Histological Type', ['pleiomorphic leiomyosarcoma', 'malignant solitary fibrous tumor', 'sclerosing epithelioid fibrosarcoma', 'myxoid fibrosarcoma', 'undifferentiated - pleiomorphic', 'synovial sarcoma', 'undifferentiated pleomorphic liposarcoma', 'epithelioid sarcoma', 'poorly differentiated synovial sarcoma', 'pleiomorphic spindle cell undifferentiated', 'pleomorphic sarcoma', 'myxofibrosarcoma', 'leiomyosarcoma'])
    mskcc_type = st.selectbox('MSKCC Type', ['MFH', 'Synovial sarcoma', 'Leiomyosarcoma'])
    site_of_primary = st.selectbox('Site of Primary STS', ['left thigh', 'right thigh', 'right parascapusular', 'left biceps', 'right buttock', 'parascapusular', 'left buttock'])
    treatment = st.selectbox('Treatment', ['Radiotherapy + Surgery', 'Radiotherapy + Surgery + Chemotherapy', 'Surgery + Chemotherapy'])
    
    user_input = {
        'Sex': sex,
        'Age': age,
        'Grade': grade,
        'Histological type': histological_type,
        'MSKCC type': mskcc_type,
        'Site of primary STS': site_of_primary,
        'Treatment': treatment
    }

    if st.button('Predict'):
        selected_model_name = st.selectbox('Choose Model', list(models.keys()))
        selected_model = models[selected_model_name]

        prediction = predict(selected_model, user_input, full_columns, scaler)
        decoded_prediction = label_encoder.inverse_transform(prediction)

        st.subheader(f'Prediction: {decoded_prediction[0]}')

        # Save the user's prediction to user records
        user_records = pd.read_csv('user_records.csv')
        user_input['Prediction'] = decoded_prediction[0]
        user_input['username'] = username  # Store the actual logged-in user
        user_records = user_records.append(user_input, ignore_index=True)
        user_records.to_csv('user_records.csv', index=False)

        st.success(f"Prediction: {decoded_prediction[0]} saved to records!")

if __name__ == '__main__':
    main_page()
