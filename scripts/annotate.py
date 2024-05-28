import cv2

# import numpy as np
import torch
from transformers import SamModel, SamProcessor


# Load SAM model and processor from Hugging Face
model = SamModel.from_pretrained("facebook/sam-vit-large")
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
# Open the video file
video_path = "../assets.snatch.mp4"
cap = cv2.VideoCapture(video_path)

# List to store barbell positions
barbell_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    inputs = processor(images=frame, return_tensors="pt").to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Get the segmentation masks
    with torch.no_grad():
        outputs = model(**inputs)

    # Assuming the barbell is the largest segmented object in the frame
    mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the barbell
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            # Calculate the center of the barbell
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            barbell_positions.append((cX, cY))

    # Optional: Display the frame with the tracked path
    for pos in barbell_positions:
        cv2.circle(frame, pos, 5, (0, 255, 0), -1)

    cv2.imshow("Barbell Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
