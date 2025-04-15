import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data import data, preprocess_data

# Make sure models folder exists
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "planet_type_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")

def encode_target_variable(data):
    """Encode planet type labels for ML models."""
    label_encoder = LabelEncoder()
    data['planet_type_encoded'] = label_encoder.fit_transform(data['planet_type'])
    joblib.dump(label_encoder, encoder_path)
    return data, label_encoder

def preprocess_features(data):
    """Prepare feature set and handle missing values."""
    X = data[['mass_earth', 'radius_earth', 'orbital_radius']]
    y = data['planet_type_encoded']
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)
    return X_scaled, scaler, y

def train_planet_type_model():
    """Train and save the planet type classification model."""
    processed_data = preprocess_data(data)
    processed_data, label_encoder = encode_target_variable(processed_data)
    X_scaled, scaler, y = preprocess_features(processed_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    return model

def predict_planet_type(mass, radius, orbital_radius):
    """Load the trained model and predict planet type from given input features."""
    # Auto-train if files are missing
    if not all(os.path.exists(path) for path in [model_path, scaler_path, encoder_path]):
        print("Model files not found. Training new model...")
        train_planet_type_model()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    input_data = np.array([[mass, radius, orbital_radius]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return label_encoder.inverse_transform(prediction)[0]
