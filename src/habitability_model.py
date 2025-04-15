import os
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Set model path
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

habitability_model_path = os.path.join(model_dir, 'habitability_model.pkl')

def identify_habitable_planets(data):
    """Define habitability criteria and add a 'habitable' column."""
    data['habitable'] = (
        (data['radius_earth'] >= 0.5) & (data['radius_earth'] <= 2.5) &
        (data['mass_earth'] >= 0.1) & (data['mass_earth'] <= 10) &
        (data['orbital_radius'] >= 0.38) & (data['orbital_radius'] <= 2.0) &
        (data['eccentricity'] <= 0.2)
    ).astype(int)
    habitable_planets = data[data['habitable'] == 1]
    return data, habitable_planets

def train_habitability_model(data):
    """Train a Decision Tree Classifier to predict habitability and save it."""
    data, _ = identify_habitable_planets(data)
    X = data[['mass_earth', 'radius_earth', 'orbital_radius', 'eccentricity']]
    y = data['habitable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train, y_train)

    y_pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Habitability Model Accuracy: {accuracy:.2f}")

    joblib.dump(dt_clf, habitability_model_path)
    return dt_clf

def predict_habitability_from_input(mass, radius, orbital_radius, eccentricity):
    """Load the trained model and predict habitability for a single input."""
    if not os.path.exists(habitability_model_path):
        from src.data import data, preprocess_data
        print("Habitability model not found. Training new one...")
        processed = preprocess_data(data)
        train_habitability_model(processed)

    model = joblib.load(habitability_model_path)
    input_df = pd.DataFrame([{
        'mass_earth': mass,
        'radius_earth': radius,
        'orbital_radius': orbital_radius,
        'eccentricity': eccentricity
    }])
    prediction = model.predict(input_df)[0]
    return bool(prediction)

def cluster_exoplanets(data, n_clusters=4):
    """Apply K-Means clustering on exoplanet features."""
    scaler = StandardScaler()
    cluster_features = data[['mass_earth', 'radius_earth', 'orbital_radius', 'eccentricity']]
    cluster_features_scaled = scaler.fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(cluster_features_scaled)
    return data, kmeans

def apply_dbscan(data):
    """Apply DBSCAN clustering and PCA for visualization."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data[['mass_earth', 'radius_earth', 'orbital_radius', 'eccentricity']])
    dbscan = DBSCAN()
    data['dbscan_cluster'] = dbscan.fit_predict(X_pca)
    return data
