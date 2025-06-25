import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model/pomegranate_disease_model_optimized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image):
    img = Image.fromarray(image.astype('uint8'), 'RGB').resize((360, 360))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    predicted_class = "Healthy" if prediction > 0.5 else "Diseased"
    return predicted_class, f"Confidence: {prediction:.2f}"

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=["text", "text"],
    title="Pomegranate Disease Detector 🍃",
    description="Upload a leaf image to check if it is healthy or diseased (Bacterial Blight)."
)

iface.launch()
