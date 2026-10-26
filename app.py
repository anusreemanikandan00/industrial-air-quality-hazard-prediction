
import gradio as gr
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


def predict_air_quality(*values):
    try:
        # Prepare input
        input_data = pd.DataFrame(
            [values], columns=features
        )

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Predict using ANN
        probabilities = model.predict(
            scaled_data, verbose=0
        )[0]

        predicted_index = np.argmax(probabilities)
        predicted_label = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        results = {
            str(label): float(probability)
            for label, probability in zip(
                label_encoder.classes_, probabilities
            )
        }

        return predicted_label, results

    except Exception as error:
        return f"Prediction error: {error}", {}


# Create input fields
inputs = [
    gr.Number(label=str(feature), value=0)
    for feature in features
]

demo = gr.Interface(
    fn=predict_air_quality,
    inputs=inputs,
    outputs=[
        gr.Textbox(label="Predicted Pollution Level"),
        gr.Label(label="Prediction Probabilities")
    ],
    title="Industrial Air Quality Hazard Prediction",
    description=(
        "Enter air-quality sensor readings to estimate "
        "the pollution category using a trained ANN model. "
        "This is an experimental prototype, not a certified "
        "industrial safety system."
    )
)

if __name__ == "__main__":
    demo.launch()
