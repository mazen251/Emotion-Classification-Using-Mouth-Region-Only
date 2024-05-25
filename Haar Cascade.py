import cv2
import numpy as np
from tensorflow.keras.models import load_model

def extract_mouth_haar(image, target_resolution=(100, 100)):
    # Load the pre-trained mouth cascade classifier
    mouth_cascade = cv2.CascadeClassifier('Weights & Models/haarcascade_mcs_mouth.xml')

    # Convert the image to grayscale for better performance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect mouths in the image
    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=15)

    # Check if at least one mouth is detected
    if len(mouths) > 0:
        # Take the first detected mouth
        (mouth_left, mouth_top, mouth_width, mouth_height) = mouths[0]

        # Extract mouth region
        mouth_region = image[mouth_top:mouth_top + mouth_height, mouth_left:mouth_left + mouth_width]

        # Draw bounding box around detected mouth
        cv2.rectangle(image, (mouth_left, mouth_top), (mouth_left + mouth_width, mouth_top + mouth_height), (0, 255, 0), 2)

        # Resize the mouth region to the target resolution
        resized_mouth = cv2.resize(mouth_region, target_resolution)

        return resized_mouth, True, None
    else:
        # Display "Mouth not detected" in red at the top right corner
        cv2.putText(image, "Mouth not detected", (image.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return None, False, 0  # Return 0 for predicted_class when mouth is not detected

# Load the trained model
model = load_model('Weights & Models/binary_emotion_model (100x50) (New Model).h5')

# Define the categories
categories = ['happy', 'sad']

# Open a video capture object (0 corresponds to the default camera, you can change it to the desired camera index)
cap = cv2.VideoCapture(0)

# Initialize variables for counters and previous predicted class
happy_count = 0
sad_count = 0
prev_predicted_class = None  # Initialize predicted_class outside the loop

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Extract mouth region using Haar Cascade
    mouth, is_detected, predicted_class = extract_mouth_haar(frame)

    if is_detected:
        # Resize the mouth region to the target resolution
        resized_mouth = cv2.resize(mouth, (100, 100))

        # Preprocess the mouth region for prediction
        resized_mouth = resized_mouth / 255.0
        resized_mouth = np.reshape(resized_mouth, (1, 100, 100, 3))  # Reshape to match the model's expected input shape

        # Make predictions
        prediction = model.predict(resized_mouth)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]  # Probability score of the predicted class

        # Update emotion counters only when the predicted class changes
        if prev_predicted_class is not None and prev_predicted_class != predicted_class:
            if categories[predicted_class] == 'happy':
                happy_count += 1
            elif categories[predicted_class] == 'sad':
                sad_count += 1

        # Display the frame with the predicted emotion and confidence
        cv2.putText(frame, f"Predicted Emotion: {categories[predicted_class]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the count of happy and sad emotions in the left bottom corner
    cv2.putText(frame, f"Happy Count: {happy_count}", (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Sad Count: {sad_count}", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Haar Cascade Emotion Prediction', frame)

    # Update the previous predicted class
    prev_predicted_class = predicted_class

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
