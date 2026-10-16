
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model and preprocessing files
model = tf.keras.models.load_model(
    "saved_model/air_quality_ann.keras"
)
scaler = joblib.load("saved_model/scaler.pkl")
label_encoder = joblib.load("saved_model/label_encoder.pkl")
features = joblib.load("saved_model/features.pkl")

# Page configuration
st.set_page_config(
    page_title="Air Quality Hazard Prediction",
    page_icon="🌫️",
    layout="wide"
)

# Dashboard heading
st.title("🌫️ Industrial Air Quality Hazard Prediction")
st.write(
    "An ANN-based prototype for classifying air-quality readings."
)

st.info(
    "Prototype only: predictions are based on experimental "
    "CO-based categories, not certified industrial safety limits."
)

st.subheader("Enter Sensor Readings")

# Input fields
input_values = {}

for feature in features:
    input_values[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.3f"
    )

# Prediction button
if st.button("Predict Pollution Level"):
    input_df = pd.DataFrame([input_values], columns=features)

    scaled_input = scaler.transform(input_df)
    probabilities = model.predict(scaled_input, verbose=0)

    predicted_index = int(np.argmax(probabilities, axis=1)[0])
    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Category: {predicted_label}")

    st.write("Model output probabilities:")
    probability_df = pd.DataFrame({
        "Category": label_encoder.classes_,
        "Probability (%)": probabilities[0] * 100
    })

    st.dataframe(probability_df, hide_index=True)
