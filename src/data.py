import pandas as pd
import numpy as np
from io import StringIO

"""Load and read the dataset."""
file_path='dataset/exoplanet_data.csv'
data = pd.read_csv(file_path)


def get_summary(data):
    """Return dataset summary as dictionary for use in templates."""

    # Raw dataset info
    buffer = StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    head = data.head()

    # Columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    missing_values = data.isnull().sum().to_dict()

    # Preprocessed dataset
    cleaned_data = preprocess_data(data.copy())
    cleaned_head = cleaned_data.head()

    return {
        'info': info_str,
        'head': head,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'missing_values': missing_values,
        'cleaned_head': cleaned_head
    }


"""Drop unnecessary columns and fill missing values."""
data.drop(columns=["stellar_magnitude"], inplace=True)
data["distance"].fillna(data["distance"].median(), inplace=True)


def estimate_orbital_radius(data):
    """Estimate missing orbital radius using Kepler’s Third Law."""
    known_data = data.dropna(subset=["orbital_radius", "orbital_period"])
    C = np.median((known_data["orbital_period"] ** 2) / (known_data["orbital_radius"] ** 3))
    
    def calculate(row):
        if np.isnan(row["orbital_radius"]):
            return ((row["orbital_period"] ** 2) / C) ** (1/3)
        return row["orbital_radius"]
    
    data["orbital_radius"] = data.apply(calculate, axis=1)
    return data

def fill_missing_multipliers(data):
    """Fill missing mass and radius multipliers using median per planet type."""
    for col in ["mass_multiplier", "radius_multiplier"]:
        data[col] = data.groupby("planet_type")[col].transform(lambda x: x.fillna(x.median()))
    return data

def standardize_units(data):
    """Convert mass and radius to Earth units."""
    def calc_mass_earth(x):
        if x["mass_wrt"] == "Earth":
            return x["mass_multiplier"]
        elif x["mass_wrt"] == "Jupiter":
            return x["mass_multiplier"] * 317.82838
    
    def calc_radius_earth(x):
        if x["radius_wrt"] == "Earth":
            return x["radius_multiplier"]
        elif x["radius_wrt"] == "Jupiter":
            return x["radius_multiplier"] * 11.209
    
    data["mass_earth"] = data.apply(calc_mass_earth, axis=1)
    data["radius_earth"] = data.apply(calc_radius_earth, axis=1)
    
    data.drop(columns=["mass_wrt", "radius_wrt"], inplace=True)
    return data

def preprocess_data(data):
    """Run all preprocessing steps."""
    data = estimate_orbital_radius(data)
    data = fill_missing_multipliers(data)
    data = standardize_units(data)
    print(data.isnull().sum())  # Check for remaining null values
    return data
