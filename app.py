from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import onnxruntime as rt
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the dataset
file_path = 'pitstops.csv'  # Change this to the path where your file is located
pitstop_data = pd.read_csv(file_path)

# Data cleaning to handle mixed data types in the 'Car' column
pitstop_data['Car'] = pitstop_data['Car'].astype(str)
pitstop_data.dropna(subset=['Car'], inplace=True)  # Remove any rows where 'Car' is NaN

# Extract unique values for dropdowns
years = sorted(pitstop_data['Year'].dropna().unique())
time_of_day_options = sorted(pitstop_data['TimeOfDay'].dropna().unique())
cars = sorted(pitstop_data['Car'].unique())

# Prepare OneHotEncoder for categorical features with fixed categories
time_of_day_categories = sorted(pitstop_data['TimeOfDay'].unique())
car_categories = sorted(pitstop_data['Car'].unique())

encoder = OneHotEncoder(categories=[time_of_day_categories, car_categories], handle_unknown='ignore')
encoder.fit(pitstop_data[['TimeOfDay', 'Car']])

# Check the number of features produced by the encoder
encoded_sample = encoder.transform([['12:04:45', 'Arrows']]).toarray()
print(f"Number of features produced by encoder: {encoded_sample.shape[1]}")

# Load ONNX models
duration_model_path = 'models/pit_stop_duration_model.onnx'
necessity_model_path = 'models/pit_stop_necessity_model.onnx'
duration_sess = rt.InferenceSession(duration_model_path)
necessity_sess = rt.InferenceSession(necessity_model_path)

# Debugging: Print the expected input dimensions
duration_input_name = duration_sess.get_inputs()[0].name
duration_input_shape = duration_sess.get_inputs()[0].shape
print(f"Duration model expected input shape: {duration_input_shape}")

necessity_input_name = necessity_sess.get_inputs()[0].name
necessity_input_shape = necessity_sess.get_inputs()[0].shape
print(f"Necessity model expected input shape: {necessity_input_shape}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_duration', methods=['GET', 'POST'])
def predict_duration():
    if request.method == 'POST':
        # Collect input features from the form
        stops = int(request.form['stops'])
        year = int(request.form['year'])
        lap = int(request.form['lap'])
        time_of_day = request.form['time_of_day']
        car = request.form['car']

        # Create input feature array
        input_features = np.array([[stops, year, lap, time_of_day, car]])

        # Encode categorical features
        categorical_features = encoder.transform(input_features[:, [3, 4]]).toarray()
        input_features = np.concatenate([input_features[:, :3], categorical_features], axis=1)

        # Debugging: Print the shape of the input features
        print(f"Input features shape: {input_features.shape}")
        print(f"Input features: {input_features}")

        # Ensure input shape matches model expectations
        if input_features.shape[1] != duration_input_shape[1]:
            return f"Input shape mismatch. Expected {duration_input_shape[1]}, but got {input_features.shape[1]}"

        # Run the model prediction
        label_name = duration_sess.get_outputs()[0].name
        duration_pred = duration_sess.run([label_name], {duration_input_name: input_features.astype(np.float32)})[0]
        
        return render_template('predict_duration.html', prediction=duration_pred[0], years=years, time_of_day_options=time_of_day_options, cars=cars)
    return render_template('predict_duration.html', years=years, time_of_day_options=time_of_day_options, cars=cars)

@app.route('/predict_necessity', methods=['GET', 'POST'])
def predict_necessity():
    if request.method == 'POST':
        # Collect input features from the form
        stops = int(request.form['stops'])
        year = int(request.form['year'])
        lap = int(request.form['lap'])
        time_of_day = request.form['time_of_day']
        car = request.form['car']

        # Create input feature array
        input_features = np.array([[stops, year, lap, time_of_day, car]])

        # Encode categorical features
        categorical_features = encoder.transform(input_features[:, [3, 4]]).toarray()
        input_features = np.concatenate([input_features[:, :3], categorical_features], axis=1)

        # Debugging: Print the shape of the input features
        print(f"Input features shape: {input_features.shape}")
        print(f"Input features: {input_features}")

        # Ensure input shape matches model expectations
        if input_features.shape[1] != necessity_input_shape[1]:
            return f"Input shape mismatch. Expected {necessity_input_shape[1]}, but got {input_features.shape[1]}"

        # Run the model prediction
        label_name = necessity_sess.get_outputs()[0].name
        necessity_pred = necessity_sess.run([label_name], {necessity_input_name: input_features.astype(np.float32)})[0]
        
        return render_template('predict_necessity.html', prediction=necessity_pred[0], years=years, time_of_day_options=time_of_day_options, cars=cars)
    return render_template('predict_necessity.html', years=years, time_of_day_options=time_of_day_options, cars=cars)

@app.route('/lap_analysis')
def lap_analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
