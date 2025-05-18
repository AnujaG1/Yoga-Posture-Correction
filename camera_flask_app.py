from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import datetime, time
import os
import mediapipe as mp
import numpy as np
from threading import Thread
from detectPose import detectPose
from classifyPose import classifyPose, getPoseCorrection

# Flask setup
app = Flask(__name__, template_folder='./templates')
app.secret_key = 'secret_key_for_flask_flash_messages'

# Global variables
capture = False
grey = False
neg = False
face = False
switch = True
rec = False
rec_frame = None

# Make shots directory to save pics
os.makedirs('./shots', exist_ok=True)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Load face detection model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

def get_working_camera_index():
    for i in range(4):
        camera = cv2.VideoCapture(i)
        if camera.isOpened():
            camera.release()
            print(f"Camera found at index {i}")
            return i
    raise Exception("Camera is not working!")

# Camera
index = get_working_camera_index()
camera = cv2.VideoCapture(index)


def record(out):
    global rec_frame, rec
    while rec:
        time.sleep(0.05)
        if rec_frame is not None:
            out.write(rec_frame)


def detect_face(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception:
        pass
    return frame


def gen_frames():
    global capture, rec_frame
    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)

        if face:
            frame = detect_face(frame)
        if grey:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame, landmarks = detectPose(frame, pose_video, display=False)
        if landmarks:
            h, w, _ = frame.shape
            keypoints = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
            label, angles = classifyPose(keypoints)
            corrections = getPoseCorrection(label, angles)

            angle_display = [f"{joint}: {int(angle)}°" for joint, angle in angles.items()]
            color = (0, 255, 0) if label != "Unknown Pose" else (0, 0, 255)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            for i, angle in enumerate(angle_display[:6]):
                cv2.putText(frame, angle, (frame.shape[1] - 220, 30 + i * 30),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            for i, correction in enumerate(corrections[:3]):
                cv2.putText(frame, correction, (10, 70 + i * 30),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

        if capture:
            capture = False
            now = datetime.datetime.now()
            filename = f'shots/shot_{now.strftime("%Y%m%d_%H%M%S")}.png'
            cv2.imwrite(filename, frame)

        if rec:
            rec_frame = frame.copy()
            frame = cv2.putText(frame, "Recording...", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ROUTES
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        weight = request.form['weight']
        experience = request.form['experience']

        print("Signup Data Received:")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Password: {password}")
        print(f"Age: {age}")
        print(f"Weight: {weight} kg")
        print(f"Yoga Experience Level: {experience}")

        flash('Signup successful!')
        return redirect(url_for('index1'))

    return render_template('signup.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global camera, capture, grey, neg, face, rec, switch, out
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            capture = True
        elif request.form.get('grey') == 'Grey':
            grey = not grey
        elif request.form.get('open') == 'Open':
            neg = not neg  # Not used
        elif request.form.get('face') == 'Face Only':
            face = not face
            if face:
                time.sleep(2)
        elif request.form.get('start') == 'Stop/Start':
            if switch:
                camera.release()
                switch = False
            else:
                index = get_working_camera_index()
                camera = cv2.VideoCapture(index)
                switch = True
        elif request.form.get('rec') == 'Start/Stop Recording':
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                filename = f'vid_{now.strftime("%Y%m%d_%H%M%S")}.avi'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
                thread = Thread(target=record, args=[out])
                thread.start()
            else:
                out.release()
    return render_template('index.html')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()