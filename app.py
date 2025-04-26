import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Functions for Model Loading and Prediction ---
def load_models():
    try:
        models = {
            'Simple ANN': load_model('Simple ANN1_model.h5'),
            'Deep ANN': load_model('Deep ANN_model.h5'),
            'CNN': load_model('CNN_model.h5'),
            'LSTM': load_model('LSTM_model.h5')
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def load_feature_columns():
    return joblib.load('input_columns.pkl')

def load_label_encoder():
    return joblib.load('label_encoder.pkl')

def preprocess_user_input(user_input, full_columns, scaler):
    categorical_cols = ['Sex', 'Grade', 'Histological type', 'MSKCC type', 'Site of primary STS', 'Treatment']
    input_df = pd.DataFrame([user_input])
    
    input_df['Age'] = scaler.transform(input_df[['Age']])
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    input_df = input_df.reindex(columns=full_columns, fill_value=0)
    
    return input_df.astype(np.float32).values

def predict(model, user_input, full_columns, scaler):
    processed_input = preprocess_user_input(user_input, full_columns, scaler)
    prediction = model.predict(processed_input)
    return prediction

def train_model(model_type, X_train, y_train):
    if model_type == 'Simple ANN':
        model = Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    elif model_type == 'Deep ANN':
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    elif model_type == 'CNN':
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    elif model_type == 'LSTM':
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], 1)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

# --- Login System ---
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

# --- Streamlit Interface ---
def main_page():
    st.title('Medical Prediction App')
    
    # Load models and encoders
    models = load_models()
    if models is None:
        return
    full_columns = load_feature_columns()
    label_encoder = load_label_encoder()
    
    # Load scaler for Age column scaling
    scaler = joblib.load('scaler.pkl')

    st.sidebar.title('Login / Register')
    page = st.sidebar.radio('Choose Page:', ['Login', 'Register'])

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

def train_page():
    st.title('Train Model')

    # Upload dataset
    file = st.file_uploader("Upload CSV dataset", type="csv")
    
    if file is not None:
        df = pd.read_csv(file)
        st.write(df.head())
        df.drop('Patient ID', axis=1, inplace=True)
        # Preprocess dataset
        X = df.drop(columns=['Status (NED, AWD, D)',])  # Assuming 'Target' column as the label column
        y = df['Status (NED, AWD, D)']

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

        # Split dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model_type = st.selectbox('Select Model', ['Simple ANN', 'Deep ANN', 'CNN', 'LSTM'])
        
        if st.button('Train Model'):
            model = train_model(model_type, X_train, y_train)
            st.success(f'{model_type} trained successfully!')

            # Evaluate the model
            y_pred = model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(int)  # Assuming binary classification
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    page = st.sidebar.radio("Select Page", ['Main Page', 'Train Model'])
    
    if page == 'Main Page':
        main_page()
    elif page == 'Train Model':
        train_page()
