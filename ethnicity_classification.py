
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from PIL import Image
import numpy as np


# Load the ethnicity classifier model, processor, and configuration
processor = AutoImageProcessor.from_pretrained("cledoux42/Ethnicity_Test_v003")
model = AutoModelForImageClassification.from_pretrained("cledoux42/Ethnicity_Test_v003")
config = AutoConfig.from_pretrained("cledoux42/Ethnicity_Test_v003")

# Retrieve the id2label mapping
id2label = config.id2label

# Video file path
video_path = '/Users/qwu_sf/CelebV-HQ/downloaded_celebvhq/raw/vtznSlhouZc.mp4'

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Frame extraction settings
frame_rate = 30  # standard frame rate ~30 frames per second 
frame_count = 0
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract frames at regular intervals
    if frame_count % frame_rate == 0:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Preprocess the image
        inputs = processor(images=pil_image, return_tensors="pt")

        # Perform classification
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = predictions.argmax().item()
        
        # Map predicted class index to label
        predicted_label = id2label[predicted_class]
        
        # Store the result
        results.append(predicted_label)

    frame_count += 1

# Release the video capture object
cap.release()

# Analyze results
print("Frame-level classifications:", results)
