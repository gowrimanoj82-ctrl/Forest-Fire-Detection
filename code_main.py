LOCAL SERVER CODE
import os
import json
import base64
from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

app = Flask(_name_)

# Try to load the model
try:
    # Use absolute path for safety
    script_dir = os.path.dirname(os.path.abspath(_file_))
    model_path = os.path.join(script_dir, "runs", "detect", "runs", "my_fire_model5", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"CRITICAL: Model file not found at {model_path}")
        model = None
    else:
        model = YOLO(model_path)
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Global variables to store the latest data
latest_sensor_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "fire_detected": False,
    "smoke_level": 0.0,
    "timestamp": None
}

# In-memory frame storage
latest_frame = None

# Image storage
latest_image_path = os.path.join("static", "latest_detection.jpg")
os.makedirs("static", exist_ok=True)
# Create a blank image to start
blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
ret, buffer = cv2.imencode('.jpg', blank_image)
latest_frame = buffer.tobytes()
cv2.imwrite(latest_image_path, blank_image)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sensor", methods=["POST"])
def sensor_endpoint():
    global latest_sensor_data
    try:
        data = request.json
        if data:
            latest_sensor_data.update(data)
            latest_sensor_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return jsonify({"status": "success", "message": "Sensor data updated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    return jsonify({"status": "error", "message": "Invalid JSON"}), 400

@app.route("/api/sensor", methods=["GET"])
def get_sensor_data():
    return jsonify(latest_sensor_data)

def gen_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        # Small delay to prevent high CPU usage
        import time
        time.sleep(0.03) 

@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files["image"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Read image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Invalid image format"}), 400

            # Run inference
            if model:
                # Use verbose=False to reduce console lag
                results = model.predict(source=img, save=False, conf=0.25, verbose=False)
                # Plot detections
                annotated_frame = results[0].plot()
            else:
                annotated_frame = img

            # Encode to JPEG in memory for streaming
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                global latest_frame
                latest_frame = buffer.tobytes()

            # Save latest detection (Optional, commenting out to increase speed)
            # cv2.imwrite(latest_image_path, annotated_frame)
            return jsonify({"status": "success", "message": "Image processed"}), 200
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)


TRAIN CODE 
import ssl
import urllib.request
from ultralytics import YOLO

# Tell Python to ignore SSL security certificate errors
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    # YOU WERE MISSING THIS LINE! This loads the AI.
    model = YOLO("yolov8n.yaml")

    print("Beginning training...")

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs",
        name="my_fire_model",
        device=0,
        amp=False  # <-- ADDED THIS: Disables the hanging AMP check!
    )

    print("Training finished!")


if _name_ == '_main_':
    main()
