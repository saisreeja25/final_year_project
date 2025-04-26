import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GlobalMaxPooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Model Building Functions ---
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

# --- Train Page Function ---
def train_page():
    st.title('Train Model')

    # Upload dataset
    file = st.file_uploader("Upload CSV dataset", type="csv")
    
    if file is not None:
        df = pd.read_csv(file)
        st.write("Dataset preview:")
        st.write(df.head())

        # Preprocess dataset
        if 'Patient ID' in df.columns:
            df.drop('Patient ID', axis=1, inplace=True)

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
        st.session_state.scaler = scaler  # Save scaler in session state

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

# --- Prediction Function ---
def predict(model, user_input, full_columns, scaler):
    # Prepare the user input for prediction
    input_data = pd.DataFrame([user_input], columns=full_columns)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

# --- Main Page Function ---
def main_page():
    st.title('Medical Prediction App')

    if 'scaler' not in st.session_state:
        st.session_state.scaler = None  # Initialize scaler if not set yet

    # Sidebar to navigate to the Train Page and Prediction
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Choose Page:', ['Train Model', 'Make Prediction'])

    if page == 'Train Model':
        train_page()  # Navigate to train model page
    elif page == 'Make Prediction':
        st.header('Enter Patient Information for Prediction')

        # User input form for prediction
        if 'scaler' in st.session_state and st.session_state.scaler is not None:
            sex = st.selectbox('Sex', ['Male', 'Female'])
            age = st.number_input('Age', min_value=0, max_value=120)
            grade = st.selectbox('Grade', ['Intermediate', 'High'])
            histological_type = st.selectbox('Histological Type', ['pleiomorphic leiomyosarcoma', 'malignant solitary fibrous tumor', 'sclerosing epithelioid fibrosarcoma', 'myxoid fibrosarcoma', 'undifferentiated - pleiomorphic', 'synovial sarcoma', 'undifferentiated pleomorphic', 'epithelioid sarcoma', 'poorly differentiated synovial sarcoma', 'pleiomorphic spindle cell undifferentiated', 'pleomorphic sarcoma', 'myxofibrosarcoma', 'leiomyosarcoma'])
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

            model_choice = st.selectbox('Choose Trained Model', ['Simple ANN', 'CNN', 'LSTM'])

            if st.button('Predict'):
                model = load_model(f'{model_choice}_model.h5')

                # Assuming the columns are defined during training
                full_columns = ['Sex', 'Age', 'Grade', 'Histological type', 'MSKCC type', 'Site of primary STS', 'Treatment']

                prediction = predict(model, user_input, full_columns, st.session_state.scaler)

                # Show the prediction result
                st.subheader(f'Prediction: {prediction}')

if __name__ == '__main__':
    main_page()
