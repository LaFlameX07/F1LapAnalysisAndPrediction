# Formula One Pit Stop Prediction and Analysis
## Project Overview
This project focuses on predicting the duration and necessity of Formula One pit stops using machine learning models. The goal is to provide teams with insights to optimize their pit stop strategies, ultimately improving race performance. The project leverages historical race data and employs Random Forest models for real-time predictions.

## Problem Statement
Pit stops are a critical aspect of Formula One racing, significantly impacting the outcome of a race. Optimizing the timing and duration of pit stops can give teams a competitive edge. However, predicting pit stop duration and necessity is challenging due to the dynamic and complex nature of races. This project aims to address this challenge by:

* ### Predicting Pit Stop Duration: 
Estimating the time required for a pit stop based on various factors.
- ### Predicting Pit Stop Necessity: 
Determining whether a pit stop is necessary at a given lap based on the current race conditions.

## Approach
* ### Data Collection and Preprocessing
1. **Data Source:** The project uses a dataset containing historical pit stop data (pitstops.csv).
2. **Data Cleaning:** Handling missing values, converting data types, and ensuring consistency in categorical values.
3. **Feature Engineering:** Creating relevant features such as the number of stops, year, lap, time of day, and car type.
4. **Model Training:** 
  - Pit Stop Duration Prediction: A RandomForestRegressor model is trained to predict the duration of pit stops.
  - Pit Stop Necessity Prediction: A RandomForestClassifier model is trained to predict whether a pit stop is necessary.
5. **Encoding Categorical Features:** Categorical features are converted to numerical using OneHotEncoder for compatibility with the models.
6. **Model Export:** Trained models are saved using joblib for deployment.
    
**More about models used:**
+ Random Forest Regressor for predicting pit stop duration
+ Random Forest Classifier for predicting pit stop necessity
* **Random Forest Regressor:**
  
  The Random Forest Regressor is used to predict the duration of pit stops. It is an ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of the predictions. The model is trained on the features (Stops, Year, Lap, TimeOfDay, Car) and the target variable (Time).
  The Random Forest Regressor is implemented using the RandomForestRegressor class from the sklearn.ensemble module. The model is trained using the fit() method, and the predictions are made using the predict() method.
- **Random Forest Classifier:**
  
  The Random Forest Classifier is used to predict whether a pit stop is necessary or not. It is also an ensemble learning method that combines multiple decision trees. The model is trained on the same features as the Random Forest Regressor, but the target variable is a binary indicator (0 for not necessary, 1 for necessary).
  The Random Forest Classifier is implemented using the RandomForestClassifier class from the sklearn.ensemble module. The model is trained using the fit() method, and the predictions are made using the predict() method.
  Both models are trained using the train_test_split() function from sklearn.model_selection to split the data into training and testing sets. The performance of the models is evaluated using the mean_squared_error() function for the regressor and the accuracy_score() function for the classifier, both from sklearn.metrics.
Finally, the trained models are saved using joblib.dump() and converted to ONNX format using skl2onnx.convert_sklearn() for deployment and inference.

### Deployment
7. **Flask Application:** A Flask web application serves as the user interface, allowing users to input race conditions and get predictions.
8. **Model Loading:** The trained models are loaded from joblib files for efficient inference.
## Lap Analysis Visualizations
The project includes several visualizations to provide insights into pit stop patterns:
1. **Average pit stop time per lap**
    This part of the code calculates the average pit stop time for each lap and plots it using a line plot. It helps to understand how the average pit stop time varies across different laps.
<p align="center">
  <img src="/static/images/lap_analysis.png" title="Average pit stop time per lap">
</p>

2. **Distribution of Pit Stop Times**
    This part of the code plots a histogram with a kernel density estimation (KDE) curve to visualize the distribution of pit stop times. It provides insights into the spread and shape of the pit stop time data.
<p align="center">
  <img src="/static/images/pit_stop_time_distribution.png" title="Distribution of Pit Stop Times">
</p>

3. **Number of Pit Stops per Year**
    This part of the code calculates the number of pit stops for each year and plots a bar chart. It helps to understand the trend in the number of pit stops over the years.
<p align="center">
  <img src="/static/images/pit_stops_per_year.png" title="Number of Pit Stops per Year">
</p>

4. **Average Pit Stop Time by Car**
    This part of the code calculates the average pit stop time for each car and plots a bar chart. It helps to compare the average pit stop time across different cars.
<p align="center">
  <img src="/static/images/average_pit_stop_time_by_car.png" title="Average Pit Stop Time by Car">
</p>
   
## Future Work
- Integrate additional features to improve model accuracy.
- Incorporate real-time data for dynamic predictions during races.
- Enhance the web application with more interactive and visually appealing elements.








# Made by Amit Murkalmath
