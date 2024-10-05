from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
from geopy.distance import geodesic
import datetime
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

app = Flask(__name__, template_folder='my_templates')
CORS(app)

TARGET_LOCATION = (13.0081912, 80.2373914)
RADIUS = 10000  # in meters
CONFIDENCE_THRESHOLD = 0.9  # Threshold for face matching

known_face_encodings = []
known_face_ids = []
# List of users with their image paths and IDs
users = [
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//jeffrey.jpg", "user_id": "Jeffrey"},
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//varsha.jpg", "user_id": "Varsha"},
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//ricky.jpg", "user_id": "Ricky"},
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//subha.jpg", "user_id": "Subha"},
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//vasanthsir.jpg", "user_id": "Vasanthan"},
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//dad.jpg", "user_id": "John"},
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//kingston.jpg", "user_id": "Kingston"},
    {"image_path": "C://Users//JEFFREY//OneDrive//Pictures//Camera Roll//sir.jpg", "user_id": "Sir"}
]

# Initialize the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

def load_known_faces():
    for user in users:
        print(f"Loading image for user: {user['user_id']} from path: {user['image_path']}")
        known_face = cv2.imread(user["image_path"])
        if known_face is None:
            print(f"Error: Unable to load image from path {user['image_path']}")
            continue
        known_face_encoding = face_recognition_embedding(known_face)
        if known_face_encoding is not None:
            known_face_encodings.append(known_face_encoding)
            known_face_ids.append(user["user_id"])
            print(f"Successfully loaded and encoded face for user: {user['user_id']}")
        else:
            print(f"Error: Failed to encode face for user: {user['user_id']}")

def is_within_radius(user_location, target_location, radius):
    distance = geodesic(user_location, target_location).meters
    print(distance)
    return distance <= radius

def log_attendance(user_id, user_location):
    with open("attendance_log.txt", "a") as file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{user_id},{timestamp},{user_location[0]},{user_location[1]}\n")
    print(f"Attendance logged for user {user_id} at {timestamp}")

def face_recognition_embedding(face_image):
    if face_image is None or face_image.size == 0:
        print("Error: Empty face image provided.")
        return None

    # Convert the image to RGB
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # Resize and preprocess the image
    face_image_rgb = cv2.resize(face_image_rgb, (160, 160))
    face_image_rgb = torch.tensor(face_image_rgb, dtype=torch.float32)
    face_image_rgb = face_image_rgb.permute(2, 0, 1) / 255.0  # Change to CxHxW and normalize to [0, 1]
    
    transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    face_image_rgb = transform(face_image_rgb)
    
    face_image_rgb = face_image_rgb.unsqueeze(0)  # Add batch dimension

    # Extract face embedding
    with torch.no_grad():
        embedding = model(face_image_rgb)
    
    return embedding.squeeze().numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log_location', methods=['POST'])
def log_location():
    data = request.get_json()
    latitude = data['latitude']
    longitude = data['longitude']
    user_location = (latitude, longitude)
    user_id = data.get('user_id', 'unknown_user')  # Replace with actual user ID logic
    print("The coordinates are", latitude, longitude)
    if is_within_radius(user_location, TARGET_LOCATION, RADIUS):
        return jsonify({"status": "success", "message": "Location verified."}), 200
    else:
        return jsonify({"status": "fail", "message": "User is not within the attendance radius."}), 400

@app.route('/verify_face', methods=['POST'])
def verify_face():
    data = request.get_json()
    image_data = data['image']

    # Decode the image
    image_data = image_data.split(",")[1]
    image_data = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"status": "fail", "message": "Image decoding failed."}), 400

    # Find face encoding in the uploaded image
    face_encoding = face_recognition_embedding(img)

    if face_encoding is None:
        return jsonify({"status": "fail", "message": "No face found in the image."}), 400
    # Check if the face matches any known face encodings
    matches = [np.linalg.norm(face_encoding - known_face_encoding) < CONFIDENCE_THRESHOLD for known_face_encoding in known_face_encodings]
    if any(matches):
        matched_idx = matches.index(True)
        matched_user_id = known_face_ids[matched_idx]
        log_attendance(matched_user_id, TARGET_LOCATION)
        return jsonify({"status": "success", "message": "Face verified and attendance logged.", "user_id": matched_user_id}), 200

    return jsonify({"status": "fail", "message": "Face not recognized."}), 400

if __name__ == '__main__':
    load_known_faces()
    app.run(host='0.0.0.0', port=5000, debug=True)
