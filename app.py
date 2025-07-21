# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
# Make sure model.pkl is in the same directory as app.py
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first.")
    model = None # Handle case where model isn't loaded

# Define the species mapping
species_mapping = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure model.pkl exists.'}), 500

    try:
        # Get data from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create a NumPy array from the input features
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(features)[0]
        predicted_species = species_mapping.get(prediction, 'Unknown')

        return render_template('index.html', prediction_text=f'Predicted Iris Species: {predicted_species}')

    except ValueError:
        return render_template('index.html', prediction_text='Error: Please enter valid numbers for all fields.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'An error occurred: {e}')

if __name__ == '__main__':
    app.run(debug=True) # Run in debug mode for development (reloads on code changes)