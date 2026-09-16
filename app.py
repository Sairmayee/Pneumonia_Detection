import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
deployment_model = load_model(
    "efficientnet_final.keras",
    compile=False
)



def predict_pneumonia(image):

    image = image.convert("L")
    image = image.resize((224, 224))

    image = np.array(image, dtype=np.float32) / 255.0

    image = np.expand_dims(image, axis=-1)

    image = tf.image.grayscale_to_rgb(
        tf.convert_to_tensor(image)
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    prediction = float(
        deployment_model.predict(
            image,
            verbose=0
        )[0][0]
    )

    return {
        "No Pneumonia": 1 - prediction,
        "Pneumonia": prediction
    }


demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(
        type="pil",
        label="Upload Chest X-Ray"
    ),
    outputs=gr.Label(
        num_top_classes=2,
        label="Prediction"
    ),
    title="Pneumonia Detection using Deep Learning",
    description="Upload a chest X-ray image to predict the probability of pneumonia."
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
