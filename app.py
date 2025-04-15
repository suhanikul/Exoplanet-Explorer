from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

# Ensure static directory exists
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

# ROUTES

@app.route('/')
def index():
    from src.data import data, get_summary

    summary = get_summary(data)

    return render_template('index.html', summary=summary)


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict_type', methods=['GET', 'POST'])
def predict_type():
    if request.method == 'POST':
        # Extract form inputs and pass to predict function
        mass = float(request.form['mass'])
        radius = float(request.form['radius'])
        orbital_radius = float(request.form['orbital_radius'])

        from src.planet_type_model import predict_planet_type
        prediction = predict_planet_type(mass, radius, orbital_radius)
        return render_template('predict_type.html', prediction=prediction)

    return render_template('predict_type.html', prediction=None)

@app.route('/predict_habitability', methods=['GET', 'POST'])
def predict_habitability():
    if request.method == 'POST':
        # Similar logic for habitability input
        mass = float(request.form['mass'])
        radius = float(request.form['radius'])
        orbital_radius = float(request.form['orbital_radius'])
        eccentricity = float(request.form['eccentricity'])

        from src.habitability_model import identify_habitable_planets
        import pandas as pd

        input_df = pd.DataFrame([{
            'mass_earth': mass,
            'radius_earth': radius,
            'orbital_radius': orbital_radius,
            'eccentricity': eccentricity
        }])

        processed_df, habitable_df = identify_habitable_planets(input_df)
        is_habitable = processed_df['habitable'].iloc[0]
        return render_template('predict_habitability.html', result=is_habitable)

    return render_template('predict_habitability.html', result=None)

@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

