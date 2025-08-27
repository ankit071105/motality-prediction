# Health Risk Assessment Tool

An AI-powered application that assesses cardiovascular health risks using machine learning and provides personalized recommendations.

## Features

- Cardiovascular risk prediction using machine learning
- Interactive health assessment form
- Visual risk score display with gauges and charts
- AI-powered health insights using Gemini AI
- Personalized health recommendations

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
4. Run the training script: `python train_model.py`
5. Launch the application: `streamlit run app.py`

## Data

The application uses cardiovascular disease data for training the model. The dataset includes:
- Age, gender, height, weight
- Blood pressure measurements
- Cholesterol and glucose levels
- Smoking and alcohol consumption
- Physical activity

## Model

The machine learning model is a Random Forest classifier trained to predict cardiovascular disease risk with over 90% accuracy.

## Disclaimer

This tool provides statistical predictions based on medical data patterns, not medical diagnoses. It should not be used as a substitute for professional medical advice.
