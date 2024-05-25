import cv2
import numpy as np
from tensorflow.keras.models import load_model

def extract_mouth_haar(image, target_resolution=(100, 100)):
    mouth_cascade = cv2.CascadeClassifier('Weights & Models/haarcascade_mcs_mouth.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=15)

    if len(mouths) > 0:
        (mouth_left, mouth_top, mouth_width, mouth_height) = mouths[0]

        mouth_region = image[mouth_top:mouth_top + mouth_height, mouth_left:mouth_left + mouth_width]

        cv2.rectangle(image, (mouth_left, mouth_top), (mouth_left + mouth_width, mouth_top + mouth_height), (0, 255, 0), 2)

        resized_mouth = cv2.resize(mouth_region, target_resolution)

        return resized_mouth, True, None
    else:
        cv2.putText(image, "Mouth not detected", (image.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return None, False, 0

model = load_model('Weights & Models/binary_emotion_model (100x50) (New Model).h5')

categories = ['happy', 'sad']

cap = cv2.VideoCapture(0)

happy_count = 0
sad_count = 0
prev_predicted_class = None

while True:
    ret, frame = cap.read()

    mouth, is_detected, predicted_class = extract_mouth_haar(frame)

    if is_detected:
        resized_mouth = cv2.resize(mouth, (100, 100))

        resized_mouth = resized_mouth / 255.0
        resized_mouth = np.reshape(resized_mouth, (1, 100, 100, 3))

        prediction = model.predict(resized_mouth)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        if prev_predicted_class is not None and prev_predicted_class != predicted_class:
            if categories[predicted_class] == 'happy':
                happy_count += 1
            elif categories[predicted_class] == 'sad':
                sad_count += 1

        cv2.putText(frame, f"Predicted Emotion: {categories[predicted_class]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Happy Count: {happy_count}", (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Sad Count: {sad_count}", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Real-Time Haar Cascade Emotion Prediction', frame)

    prev_predicted_class = predicted_class

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
