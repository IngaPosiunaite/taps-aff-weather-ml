“Taps aff” is a Scottish phrase meaning “tops off”, commonly used when the weather is unusually warm.

This project uses machine learning to predict whether a day in Glasgow is a “taps aff” day based on historical weather data.

---

## Project Overview

The project combines **regression** and **binary classification** to:
- Estimate missing weather values
- Predict warm “taps aff” days from 2023 to 2025

It demonstrates a full machine learning workflow, including data cleaning, feature engineering, model training, and evaluation.

---

## What This Project Does

### 1. Handle Missing Weather Data
A regression model is trained to estimate missing values (such as daylight duration) using complete weather records.

### 2. Feature Engineering
Dates are transformed into useful features (day and month) to help models learn seasonal patterns.

### 3. Merge Datasets
Weather data is combined with a labeled “taps aff” dataset to prepare training data for classification.

### 4. Predict “Taps Aff” Days
A binary classification deep learning model predicts whether a given day qualifies as a warm “taps aff” day.

### 5. Evaluate Performance
Model predictions are evaluated using test data and visualized to confirm realistic seasonal trends.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- TensorFlow / Keras  
- Scikit-learn  

---

## What I Learned

- How to handle missing data using regression  
- Feature engineering for time-based patterns  
- Training deep learning models for classification  
- Evaluating model performance on test datasets  
- Building an end-to-end machine learning pipeline  

