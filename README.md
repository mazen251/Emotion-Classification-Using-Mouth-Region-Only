# Emotion Classification Using Mouth Region Only
Welcome to the Emotion Classification Using Mouth Region Only project! This repository contains code to detect the mouth region of the face and classify emotions (happy or sad) using a deep learning model. The primary contribution of this project is that it doesn't rely on any face detection algorithms to run its emotion classification techniques. This makes it ideal for scenarios where the face is not completely visible, and other emotion classification algorithms fail due to the failure of face detection at first.

## Table of Contents
- Introduction
- Features
- Demo
- Installation
- Usage
- Database and Pre-trained Models
- Contributing
- License
- Contact
- Acknowledgements

## Introduction
The Emotion Classification Using Mouth Region Only project aims to detect the mouth region and classify emotions based solely on the mouth. The face can be partially visible, with the eyes, eyebrows, nose, and the whole upper face region not being important. This makes it ideal for scenarios where the face is not fully visible, where other emotion classification algorithms fail due to the failure of face detection.

## Features
- **YOLOv8 Model:** Uses the state-of-the-art YOLOv8 model for detecting faces.
- **Haar Cascade:** Utilizes Haar Cascade specifically trained to focus on the mouth region.
- **Combination:** Combines YOLOv8 and Haar Cascade for robust mouth detection and emotion classification.

## Demo
![Image Example](Demos/Run.jpg)

## Installation
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites:
- Python 3.x
- OpenCV
- TensorFlow
- NumPy

### Steps
```sh
#Clone the Repository
git clone https://github.com/mazen251/Emotion-Classification-Using-Mouth-Region-Only.git
cd Emotion-Classification-Using-Mouth-Region-Only

#Install Dependencies
pip install -r requirements.txt

#Prepare the Model
Place your trained TensorFlow model (.h5 file) in the Weights & Models directory. Or use the existing CNN model that i have trained.

#Run the App any of the .py files you want.
```
## Usage
You can run any .py file in this repository either on a real-time video camera feed or a pre-recorded video. This can be changed in the code by setting the appropriate video source.
### Real-time Video Camera Feed:
- When running on a real-time video feed, the application will save the captured images in the Cropped Images from Run folder.

### Pre-recorded Video:
- When running on a pre-recorded video, the processed video will be saved in the main directory.

### View Results:
- The application will display the video with detected mouth regions and classified emotions or save the images directly to the Cropped Images from Run folder.

## Database and Pre-trained Models
The Weights & Models folder contains the following:

- YOLOv8 Model: A pre-trained YOLOv8 model (yolov8n-face.onnx) for detecting faces.
- Haar Cascade: An XML file for Haar Cascade trained specifically to detect mouth regions (haarcascade_mcs_mouth.xml).
- Pre-trained CNN Classifier: A CNN model (binary_emotion_model.h5) trained on a custom-made mouth region dataset. This model captures the cropped ROI image from YOLOv8 and Haar Cascade (mouth region) and classifies the emotion as happy or sad.
- 
A custom-made mouth region dataset will be provided on Kaggle soon. Stay tuned for updates!

## Contributing
contributions are welcomed to improve this project. To contribute, please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Commit your changes (git commit -m 'Add some feature').
- Push to the branch (git push origin feature/your-feature).
- Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` for more details.

## Contact
For any inquiries or further information, please contact:

Mazen Walid - [@Mazen Walid](https://www.linkedin.com/in/mazen-walid-225582208/)

## Acknowledgements
- TensorFlow
- YOLO
- OpenCV
- Haar Cascade

