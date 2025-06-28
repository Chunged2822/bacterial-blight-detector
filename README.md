# Bacterial Blight Detector: AI for Pomegranate Leaf Health 🌱🔍

![Bacterial Blight Detector](https://img.shields.io/badge/Download%20Now-Release-brightgreen) [![GitHub Release](https://img.shields.io/github/release/Chunged2822/bacterial-blight-detector.svg)](https://github.com/Chunged2822/bacterial-blight-detector/releases)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
The **Bacterial Blight Detector** is an AI-powered web application designed to detect bacterial blight on pomegranate leaves. Using a lightweight TensorFlow Lite model, this tool offers quick and accurate results. The application helps farmers and researchers identify plant diseases early, promoting healthier crops and better yields.

You can download the latest version of the application from the [Releases section](https://github.com/Chunged2822/bacterial-blight-detector/releases). 

## Features
- **Real-time Detection**: Quickly analyze pomegranate leaves for signs of bacterial blight.
- **User-friendly Interface**: Built with Gradio for easy interaction.
- **Lightweight Model**: Utilizes TensorFlow Lite for efficient performance.
- **Cross-platform Compatibility**: Works on various devices and operating systems.
- **Open Source**: Contribute to the project and enhance its capabilities.

## Technologies Used
- **CNN**: Convolutional Neural Networks for image classification.
- **TensorFlow**: Core framework for building and training the model.
- **Keras**: Simplifies the model-building process.
- **Gradio**: Provides an easy-to-use interface for web applications.
- **Hugging Face**: Integrates pre-trained models for enhanced performance.
- **Data Science**: Employs data analysis techniques for effective training.
- **Machine Learning**: Implements algorithms for disease detection.

## Installation
To set up the Bacterial Blight Detector on your local machine, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Chunged2822/bacterial-blight-detector.git
   cd bacterial-blight-detector
   ```

2. **Install Required Packages**:
   Use pip to install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Model**:
   You can find the model files in the [Releases section](https://github.com/Chunged2822/bacterial-blight-detector/releases). Download and place them in the `model` directory.

4. **Run the Application**:
   Start the application using:
   ```bash
   python app.py
   ```

## Usage
Once the application is running, you can access it via your web browser. Follow these steps:

1. **Upload an Image**: Drag and drop or click to upload a pomegranate leaf image.
2. **Analyze the Image**: The model will process the image and provide results indicating the presence of bacterial blight.
3. **Review Results**: Check the output for information on disease severity and recommendations for treatment.

## Model Training
If you want to train your own model, follow these steps:

1. **Prepare Your Dataset**: Collect images of healthy and infected pomegranate leaves. Ensure you have a balanced dataset for effective training.
2. **Preprocess the Images**: Resize and normalize the images. Use the following script:
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   datagen = ImageDataGenerator(rescale=1./255)
   train_generator = datagen.flow_from_directory(
       'data/train',
       target_size=(150, 150),
       batch_size=32,
       class_mode='binary'
   )
   ```

3. **Build the Model**: Use a CNN architecture. Here’s a simple example:
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(1, activation='sigmoid')
   ])
   ```

4. **Compile and Train the Model**:
   ```python
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(train_generator, epochs=10)
   ```

5. **Save the Model**:
   After training, save your model for later use:
   ```python
   model.save('model/bacterial_blight_detector.h5')
   ```

## Contributing
We welcome contributions from everyone. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch and create a pull request.

Please ensure that your code follows the existing style and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or suggestions, feel free to reach out:

- **Email**: example@example.com
- **GitHub**: [Chunged2822](https://github.com/Chunged2822)

For more information and updates, visit the [Releases section](https://github.com/Chunged2822/bacterial-blight-detector/releases).