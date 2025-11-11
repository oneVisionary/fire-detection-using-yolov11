from flask import Flask, redirect, session, url_for, render_template, request, jsonify
import json
import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask_session import Session
from datetime import datetime
from collections import Counter
# List of locations
locations = [
    [8.699900284031973, 77.74306222856937],
    [8.70528778413457, 77.75456354073003],
    [8.711056989576877, 77.74018690052918],
    [8.710887308333989, 77.73224756191081],
    [8.693070349647227, 77.74254724444276]
]

# Select a random location
selected_locations = random.sample(locations, 1)

# Get current date
now = datetime.now()
current_date = now.date()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.secret_key = 'your_secret_key' 
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Define JSON data structure
json_data = {
    "date": "",
    "location": ""
}
dic = []
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/view_metrics')
def view_metrics():
    return render_template('view_performance.html')

@app.route('/index')
def index():
    session['mode'] = ""
    return render_template('index.html')

@app.route('/predict')
def predict():
    session['mode'] =""
    return render_template('predict.html')

def store_data(new_data):
    json_file_path = os.path.join("static/", 'prediction_results.json')
    
    # Read existing data
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            try:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    existing_data = []  
            except json.JSONDecodeError:
                existing_data = []  
    else:
        existing_data = []  
    
    # Append new data
    if isinstance(existing_data, list):
        existing_data.append(new_data)
    else:
        existing_data = [new_data]
    
    # Write updated data
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)  

def load_and_sort_data():
    with open('static/prediction_results.json', 'r') as file:
        data = json.load(file)

    for item in data:
        item["date"] = datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S")
    

    data.sort(key=lambda x: x["date"])
    
    return data
from ultralytics import YOLO
model = YOLO("./models/fire_detector.pt")   # Load YOLO fire model

dic = []  

def Test(video_path, result_folder):
    global dic
    dic = []  # Reset data list before processing each video

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    total_frames = 0
    fire_frames = 0
    no_fire_frames = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_index += 1

        # YOLO Prediction
        results = model(frame, verbose=False)
        fire_detected = False
        
        # Check detection results
        for result in results:
            if result.boxes and len(result.boxes) > 0:
                fire_detected = True

        # Update counters & list
        if fire_detected:
            dic.append("fire_images")
            fire_frames += 1
            print(f"Frame {frame_index}: 🔥 FIRE DETECTED")
        else:
            dic.append("non_fire_images")
            no_fire_frames += 1
            print(f"Frame {frame_index}: ✅ NO FIRE")

        # Save annotated frame
        annotated_frame = results[0].plot()
        save_path = os.path.join(result_folder, f"frame_{frame_index}.jpg")
        cv2.imwrite(save_path, annotated_frame)

    cap.release()
    cv2.destroyAllWindows()

    print("\n========== SUMMARY ==========")
    print("Total Frames:", total_frames)
    print("Frames with Fire:", fire_frames)
    print("Frames with No Fire:", no_fire_frames)

    # ✅ Determine Majority Prediction
    if len(dic) > 0:
        most_common = Counter(dic).most_common(1)[0][0]
    else:
        most_common = "non_fire_images"

    session['mode'] = most_common
    print(f"Final Prediction (Majority): {most_common}")

    # ✅ Save Summary to summary.json
    summary_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_frames": total_frames,
        "fire_frames": fire_frames,
        "no_fire_frames": no_fire_frames,
        "final_prediction": most_common
    }

    summary_path = os.path.join("static", "summary.json")
    with open(summary_path, "w") as json_file:
        json.dump(summary_data, json_file, indent=4)

    print(f"✅ Summary saved to {summary_path}")

    # ✅ If fire detected → Log event in prediction_results.json
    if most_common == "fire_images":
        alert_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": selected_locations  # already random location
        }
        store_data(alert_data)
        print("🔥 Fire alert saved successfully!")
def clear_result_folder():
    folder = app.config['RESULT_FOLDER']
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            os.remove(file_path)
        except:
            pass

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    session['predictions'] = None
    if request.method == 'POST':
        if 'videoFile' not in request.files:
            return redirect(request.url)
        file = request.files['videoFile']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            clear_result_folder()
            # Call Test function
            Test(video_path, app.config['RESULT_FOLDER'])
            
            return redirect(url_for('result'))
    return render_template('predict.html')

@app.route('/result')
def result():
    result_files = os.listdir(app.config['RESULT_FOLDER'])
    result_files = sorted(result_files)
    return render_template('predict.html', files=result_files)

@app.route('/login', methods=['GET', 'POST'])
def login(): 
    return render_template('login.html')

@app.route('/authorised', methods=['GET', 'POST'])
def authorised(): 
    email = request.form.get("email")
    pwd = request.form.get("pwd")
    if email == "admin@gmail.com" and pwd == "admin":
        data = load_and_sort_data()
        print(data)
        return render_template('admin.html' , data=data)
    else:
        return render_template('login.html', message="Sorry Invalid credentials")


@app.route('/map_view/<float:lat>/<float:lng>')
def map_view(lat, lng):
    return render_template('map.html', lat=lat, lng=lng)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
