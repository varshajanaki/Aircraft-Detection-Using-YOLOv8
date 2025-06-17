import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO('models/aircraft_yolov8s.pt')

def detect_aircraft(image):
    # Run inference
    results = model(image)
    result = results[0]
    
    # Convert image to numpy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Create a copy of the image for drawing
    annotated_image = image.copy()
    
    # Process results
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get box coordinates
        conf = float(box.conf[0])  # Get confidence score
        cls = int(box.cls[0])  # Get class index
        class_name = model.names[cls]  # Get class name
        
        # Draw box on the image
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, f"{class_name} {conf:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detections.append(f"{class_name}: {conf:.2f}")
    
    return annotated_image, "\n".join(detections)

# Create Gradio interface
demo = gr.Interface(
    fn=detect_aircraft,
    inputs=gr.Image(),
    outputs=[
        gr.Image(label="Detected Aircraft"),
        gr.Textbox(label="Detections")
    ],
    title="Military Aircraft Detection",
    description="Upload an image to detect military aircraft using YOLOv8"
)

if __name__ == "__main__":
    demo.launch()