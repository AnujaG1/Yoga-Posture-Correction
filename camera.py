# from flask import Flask, render_template, Response, request, jsonify, flash
# import cv2
# import datetime, time
# import os, sys
# import mediapipe as mp
# import numpy as np
# from threading import Thread
# from detectPose import detectPose
# from classifyPose import classifyPose, getPoseCorrection

# # Flask setup
# app = Flask(__name__, template_folder='./templates')
# app.secret_key = 'secret_key_for_flask_flash_messages'

# # Global variables
# capture = False
# grey = False
# neg = False
# face = False
# switch = True
# rec = False
# rec_frame = None

# # Make shots directory to save pics
# os.makedirs('./shots', exist_ok=True)


# # MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
# mp_drawing = mp.solutions.drawing_utils

# # Setup Pose function for video.

# camera = cv2.VideoCapture(0)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Set width
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

# def record(out):
#     global rec_frame, rec
#     while rec:
#         time.sleep(0.05)
#         if rec_frame is not None:
#             out.write(rec_frame)


# def detect_face(frame):
#     global net
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (500, 500)), 1.0,
#         (300, 300), (104.0, 177.0, 123.0))   
#     net.setInput(blob)
#     detections = net.forward()
#     confidence = detections[0, 0, 0, 2]

#     if confidence < 0.5:            
#             return frame           

#     box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
#     (startX, startY, endX, endY) = box.astype("int")
#     try:
#         frame=frame[startY:endY, startX:endX]
#         (h, w) = frame.shape[:2]
#         r = 480 / float(h)
#         dim = ( int(w * r), 480)
#         frame=cv2.resize(frame,dim)
#     except Exception as e:
#         pass
#     return frame
 

# def gen_frames():
#     global capture, rec_frame
#     while True:
#         success, frame = camera.read()
#         if not success:
#             continue

#         frame = cv2.flip(frame, 1)

#         if face:
#             frame = detect_face(frame)
#         if grey:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#         frame, landmarks = detectPose(frame, pose_video, display=False)
#         if landmarks:
#             h, w, _ = frame.shape
#             keypoints = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
#             label, angles = classifyPose(keypoints)
#             corrections = getPoseCorrection(label, angles)

#             angle_display = [f"{joint}: {int(angle)}°" for joint, angle in angles.items()]
#             color = (0, 255, 0) if label != "Unknown Pose" else (0, 0, 255)
#             cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

#             for i, angle in enumerate(angle_display[:6]):
#                 cv2.putText(frame, angle, (frame.shape[1] - 220, 30 + i * 30),
#                             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

#             for i, correction in enumerate(corrections[:3]):
#                 cv2.putText(frame, correction, (10, 70 + i * 30),
#                             cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
                
#         if capture:
#                 capture = False
#                 now = datetime.datetime.now()
#                 filename = f'shots/shot_{now.strftime("%Y%m%d_%H%M%S")}.png'
#                 cv2.imwrite(filename, frame)

#         if rec:
#             rec_frame = frame.copy()
#             frame = cv2.putText(frame, "Recording...", (10, 25),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# # ROUTES
# @app.route('/')
# def index():
#     return render_template('index1.html')

# @app.route('/index1')
# def index1():
#     return render_template('index1.html')

# @app.route('/demo')
# def demo():
#     return render_template('demo.html')


# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/blog')
# def blog():
#     return render_template('blog.html')

# @app.route("/contact")
# def contact():
#     return render_template("contact.html")


# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         age = request.form['age']
#         weight = request.form['weight']
#         experience = request.form['experience']

#         # Log user data (can be saved to database later)
#         print("Signup Data Received:")
#         print(f"Username: {username}")
#         print(f"Email: {email}")
#         print(f"Password: {password}")
#         print(f"Age: {age}")
#         print(f"Weight: {weight} kg")
#         print(f"Yoga Experience Level: {experience}")

#         flash('Signup successful!')
#         return render_template('index1')

#     return render_template('signup.html')

# @app.route('/submit_contact', methods=['POST'])
# def submit_contact():
#     name = request.form['name']
#     email = request.form['email']
#     message = request.form['message']

#     print(f"Contact received: {name}, {email}, {message}")

#     return f'''
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <title>Thank You</title>
#         <style>
#             body {{
#                 font-family: Arial, sans-serif;
#                 background-color: #f9f9f9;
#                 margin: 0;
#                 padding: 0;
#             }}
#             .thank-you-box {{
#                 max-width: 500px;
#                 margin: 100px auto;
#                 padding: 30px;
#                 background: white;
#                 border-radius: 12px;
#                 box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
#                 text-align: center;
#             }}
#             h1 {{
#                 color: #4CAF50;
#             }}
#             p {{
#                 margin-top: 10px;
#                 color: #555;
#             }}
#             a {{
#                 display: inline-block;
#                 margin-top: 20px;
#                 color: #007bff;
#                 text-decoration: none;
#                 font-weight: bold;
#             }}
#         </style>
#     </head>
#     <body>
#         <div class="thank-you-box">
#             <h1>Thank You, {name}!</h1>
#             <p>Your message has been received. We'll get back to you soon.</p>
#            <div style="text-align: center; margin: 20px;">
#         <a href="{{ url_for('index') }}" class="back-btn">← Back to Home</a>
#     </div>
#         </div>
#     </body>
#     </html>
#     '''



# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     user_message = request.json.get("message", "").lower()

#     responses = {
#         "what is yoga": "Yoga is a physical, mental, and spiritual practice that originated in ancient India.",
#         "benefits of yoga": "Yoga improves flexibility, builds strength, reduces stress, and boosts mental clarity.",
#         "how often should I do yoga": "Even 10-20 minutes daily can be beneficial!",
#         "what's the best pose for beginners": "The Mountain Pose and Child’s Pose are great for beginners.",
#         "namaste": "Namaste! How can I help you with yoga today?",
#     }

#     # Default response if question not found
#     reply = responses.get(user_message.strip(), "I'm still learning! Please ask something else about yoga.")
#     return jsonify({"reply": reply}) 

# @app.route('/requests',methods=['POST','GET'])
# def tasks():
#     global switch,camera
#     if request.method == 'POST':
#         if request.form.get('click') == 'Capture':
#             global capture
#             capture= True
#         elif  request.form.get('grey') == 'Grey':
#             global grey
#             grey=not grey
#         if  request.form.get('open') == 'Open':
#             global neg
#             neg=not neg
#         elif  request.form.get('face') == 'Face Only':
#             global face
#             face=not face 
#             if(face):
#                 time.sleep(4)   
#         elif  request.form.get('start') == 'Stop/Start':
            
#             if(switch):
                
#                 camera = cv2.VideoCapture(0)
#                 switch= False
                
                
#             else:
#                 switch= True
#                 camera.release()
#                 cv2.destroyAllWindows()
#         elif  request.form.get('rec') == 'Start/Stop Recording':
#             global rec, out
#             rec= not rec
#             if(rec):
#                 now=datetime.datetime.now() 
#                 fourcc = cv2.VideoWriter_fourcc(*'XVID')
#                 out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
#                 #Start new thread for recording the video
#                 thread = Thread(target = record, args=[out,])
#                 thread.start()
#             elif(rec==False):
#                 out.release()
            
#     elif request.method=='GET':
#         return render_template('index.html')
#     return render_template('index.html')

# if __name__ == '__main__':
#     try:
#         app.run(debug=True)
#     finally:
#         camera.release()
#         cv2.destroyAllWindows()


from flask import Flask, render_template, Response, request, jsonify, flash
import cv2
import datetime
import time
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
switch = False  # Start with camera off
rec = False
rec_frame = None
camera = None
out = None

# Make shots directory to save pics
os.makedirs('./shots', exist_ok=True)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Load DNN model for face detection
prototxt = "deploy.prototxt"  # Ensure this file is in the project directory
model = "res10_300x300_ssd_iter_140000.caffemodel"  # Ensure this file is in the project directory
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
except Exception as e:
    print(f"Error loading DNN model: {e}")
    net = None

def record(out):
    global rec_frame, rec
    while rec:
        time.sleep(0.05)
        if rec_frame is not None:
            out.write(rec_frame)
        else:
            print("Warning: No frame available for recording.")

def detect_face(frame):
    global net
    if net is None:
        print("Error: Face detection model not loaded.")
        return frame
    try:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (500, 500)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        if detections.shape[2] == 0:
            return frame
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
        except Exception as e:
            print(f"Face detection error: {e}")
        return frame
    except Exception as e:
        print(f"Error in detect_face: {e}")
        return frame

def gen_frames():
    global capture, rec_frame, switch, camera
    while True:
        if not switch or camera is None or not camera.isOpened():
            time.sleep(0.1)  # Avoid busy loop when camera is off
            continue
        success, frame = camera.read()
        if not success:
            print("Error: Failed to read frame.")
            continue
        frame = cv2.flip(frame, 1)
        if face:
            frame = detect_face(frame)
        if grey:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        try:
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
        except Exception as e:
            print(f"Error in pose detection: {e}")
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
        if not ret:
            print("Error: Failed to encode frame.")
            continue
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

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/live_classes")
def live_classes():
    return render_template("live_classes.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        age = request.form.get('age')
        weight = request.form.get('weight')
        experience = request.form.get('experience')
        # Log user data (can be saved to database later)
        print("Signup Data Received:")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Password: {password}")
        print(f"Age: {age}")
        print(f"Weight: {weight} kg")
        print(f"Yoga Experience Level: {experience}")
        flash('Signup successful!')
        return render_template('index1.html')
    return render_template('signup.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    print(f"Contact received: {name}, {email}, {message}")
    return render_template('thank_you.html', name=name)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get("message", "").lower()
    responses = {
        "what is yoga": "Yoga is a physical, mental, and spiritual practice that originated in ancient India.",
        "benefits of yoga": "Yoga improves flexibility, builds strength, reduces stress, and boosts mental clarity.",
        "how often should I do yoga": "Even 10-20 minutes daily can be beneficial!",
        "what's the best pose for beginners": "The Mountain Pose and Child’s Pose are great for beginners.",
        "namaste": "Namaste! How can I help you with yoga today?",
    }
    reply = responses.get(user_message.strip(), "I'm still learning! Please ask something else about yoga.")
    return jsonify({"reply": reply})

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, capture, grey, neg, face, rec, out
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            capture = True
        elif request.form.get('grey') == 'Grey':
            grey = not grey
        elif request.form.get('open') == 'Open':
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            face = not face
            if face:
                time.sleep(4)
        elif request.form.get('start') == 'Stop/Start':
            if switch:
                switch = False
                if camera is not None:
                    camera.release()
                    camera = None
            else:
                switch = True
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    print("Error: Could not open camera.")
                    switch = False
                    camera = None
                    flash('Failed to open camera!')
                    return render_template('index1.html')
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)
        elif request.form.get('rec') == 'Start/Stop Recording':
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(f'vid_{now.strftime("%Y%m%d_%H%M%S")}.avi', fourcc, 20.0, (1280, 750))
                thread = Thread(target=record, args=[out,])
                thread.start()
            elif out is not None:
                out.release()
                out = None
    return render_template('index1.html')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if camera is not None:
            camera.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()