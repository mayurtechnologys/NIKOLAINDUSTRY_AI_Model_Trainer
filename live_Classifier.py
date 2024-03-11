import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load your TensorFlow model and labels
model = load_model('NIKOLAINDUSTRY_model_trained.h5')
labels = {}
with open('labels.txt', 'r') as label_file:
    for line in label_file:
        index, label = line.strip().split()
        labels[int(index)] = label

# Function to preprocess the frames from the webcam
def preprocess_frame(frame):
    # Resize the frame to match the input shape expected by the model
    resized_frame = cv2.resize(frame, (32, 32))
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to be in the range [0, 1]
    normalized_frame = rgb_frame / 255.0
    # Expand dimensions to match the input shape expected by the model
    expanded_frame = np.expand_dims(normalized_frame, axis=0)
    return expanded_frame

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform inference using the model
    prediction = model.predict(processed_frame)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = labels.get(predicted_class_index, "Unknown")

    # Display the predicted class label on the frame
    cv2.putText(frame, predicted_class_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the predicted class label
    cv2.imshow('NIKOLAINDUSTRY Live Image Classifier', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
