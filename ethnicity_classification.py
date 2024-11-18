
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from PIL import Image
import numpy as np


processor = AutoImageProcessor.from_pretrained("cledoux42/Ethnicity_Test_v003")
model = AutoModelForImageClassification.from_pretrained("cledoux42/Ethnicity_Test_v003")
config = AutoConfig.from_pretrained("cledoux42/Ethnicity_Test_v003")
id2label = config.id2label

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path = '/Users/qwu_sf/CelebV-HQ/downloaded_celebvhq/raw/vtznSlhouZc.mp4'

cap = cv2.VideoCapture(video_path)

frame_rate = 30  # standard frame rate: 30 frames per second 
frame_count = 0
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_rate == 0:
        frame_results = []  # Store classifications for this frame

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop the face region
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)

            inputs = processor(images=pil_face, return_tensors="pt")

            # Classification
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = predictions.argmax().item()
                predicted_label = id2label[predicted_class]

            frame_results.append(predicted_label)

        results.append({"frame": frame_count, "classifications": frame_results})

    frame_count += 1

cap.release()

print("Frame-level classifications:", results)