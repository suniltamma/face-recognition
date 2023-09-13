from flask import Flask, render_template, request, jsonify, Response
from deepface import DeepFace
import cv2
import os
import threading
import atexit
import numpy as np

app = Flask(__name__)

# Directory to store registered faces
registered_faces_dir = 'registered_faces'

if not os.path.exists(registered_faces_dir):
    os.makedirs(registered_faces_dir)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Flag to indicate when to stop the camera thread
camera_thread_stop = False

def generate_frames():
    while True:
        if camera_thread_stop:
            break
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# Define a route to stream the camera feed for face registration
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route for face registration
@app.route('/register', methods=['POST'])
def register_face():
    try:
        # Capture a frame from the camera
        success, frame = camera.read()

        if success:
            # Perform face detection
            detected_face = DeepFace.detectFace(frame, detector_backend='opencv')

            if detected_face is not None:
                # Save the detected face for registration
                filename = os.path.join(registered_faces_dir, 'registered_face.jpg')

                # Save the image using OpenCV
                cv2.imwrite(filename, detected_face)

                return jsonify({"message": "Face registered successfully."})
            else:
                return jsonify({"error": "No face detected in the captured frame."})
        else:
            return jsonify({"error": "Failed to capture a frame from the camera."})

    except Exception as e:
        return jsonify({"error": str(e)})

# Define a route for face recognition
@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        # Get the uploaded image
        image = request.files['image'].read()

        # Load the registered face
        registered_face_path = os.path.join(registered_faces_dir, 'registered_face.jpg')

        # Read the image using OpenCV
        registered_face = cv2.imread(registered_face_path)

        if registered_face is not None:
            # Perform face recognition without enforcing detection
            result = DeepFace.verify(image, registered_face, enforce_detection=False)

            if result["verified"]:
                return jsonify({"message": "Face recognized successfully."})
            else:
                return jsonify({"error": "Face not recognized."})
        else:
            return jsonify({"error": "No registered face found."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    camera_thread = threading.Thread(target=generate_frames)
    camera_thread.start()
    app.run(debug=True, threaded=True, use_reloader=False)

# Release the camera when the app is closed
atexit.register(lambda: camera.release())
