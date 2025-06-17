from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import numpy as np
import uuid
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = YOLO('models/aircraft_yolov8s.pt')

# Get class names from the model
class_names = model.names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Generate a unique filename to prevent conflicts
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    
    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Run inference
    results = model(filepath)
    result = results[0]  # Get the first result
    
    # Process results
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get box coordinates
        conf = float(box.conf[0])  # Get confidence score
        cls = int(box.cls[0])  # Get class index
        class_name = class_names[cls]  # Get class name
        
        detections.append({
            'box': [x1, y1, x2, y2],
            'confidence': conf,
            'class': class_name
        })
    
    # Draw boxes on the image with improved visuals
    img = cv2.imread(filepath)
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['confidence']
        class_name = det['class']
        
        # Determine color based on confidence (green for high, yellow for medium, red for low)
        if conf > 0.8:
            color = (0, 255, 0)  # Green
        elif conf > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw a slightly thicker rectangle with rounded corners
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add a semi-transparent background for text
        text = f"{class_name} {conf:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
        
        # Add text with better visibility
        cv2.putText(img, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save the annotated image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{unique_filename}")
    cv2.imwrite(output_path, img)
    
    return jsonify({
        'detections': detections,
        'image_url': f"/static/uploads/detected_{unique_filename}"
    })

# Add a route to clean up old files (optional)
@app.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    # Delete files older than 1 hour
    current_time = time.time()
    deleted_count = 0
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue
        # Check file age
        file_age = current_time - os.path.getmtime(file_path)
        if file_age > 3600:  # 1 hour in seconds
            os.remove(file_path)
            deleted_count += 1
    
    return jsonify({'message': f'Deleted {deleted_count} old files'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)