# 🍃 Pomegranate Bacterial Blight Detector

This project uses a Convolutional Neural Network (CNN) model converted to **TensorFlow Lite** to detect **bacterial blight disease** on **pomegranate leaves**.  
It provides a fast, lightweight, and user-friendly web interface built with **Gradio**, deployable on **Hugging Face Spaces** or locally.

---

## 🔗 Live Demo

👉 Try the app live on Hugging Face:  
[https://huggingface.co/spaces/omkar2503/bacterial-blight-detector](https://huggingface.co/spaces/omkar2503/bacterial-blight-detector)

---

## 📥 Download the Model

The trained TensorFlow Lite model is not included in the repository due to GitHub's file size limits.

➡️ [Download `pomegranate_disease_model_optimized.tflite`](https://drive.google.com/file/d/1-82orr-CPDE8qDQaugkAWLllZypzP6rB/view?usp=sharing)

After downloading, create a folder named `model/` and place the file inside:

bacterial-blight-detector/
├── model/

│ └── pomegranate_disease_model_optimized.tflite

├── app.py

├── requirements.txt

└── README.md

pip install -r requirements.txt
python app.py
