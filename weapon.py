import torch
import cv2
from ultralytics import YOLO
import numpy as np

# Load the pre-trained model
model_path = '/Users/manojasher/Desktop/best (11).pt'
model = YOLO(model_path)
# Access webcam (default camera index is 0)
def weapon_detection_webcam():
    # Initialize webcam (0 = default webcam, adjust index for other cameras)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to exit the webcam feed.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform inference using the loaded model
        results = model.predict(source=frame, save=False, conf=0.5)  # Set confidence threshold (e.g., 0.5)

        # Visualize the results directly on the frame
        annotated_frame = results[0].plot()  # `results[0]` is the first result; `plot()` adds bounding boxes

        # Show the webcam feed with detections
        cv2.imshow("Weapon Detection", annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    weapon_detection_webcam()