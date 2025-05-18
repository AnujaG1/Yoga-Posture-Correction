import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pyttsx3
import threading
import time

# ========== Pyttsx3 Setup ==========

# Initialize pyttsx3 engine globally (once)
engine = pyttsx3.init()

# Thread-safe speak function
def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# Last time speech was triggered
last_speak_time = 0

def should_speak(interval=5):
    global last_speak_time
    current_time = time.time()
    if current_time - last_speak_time > interval:
        last_speak_time = current_time
        return True
    return False


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True,
                    min_detection_confidence=0.5, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils


def detectPose(image, pose, display=True):
    
    # Create a copy of the input image.
    output_image = image.copy()
    

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))
            
        # === Trigger speech only when landmarks are detected ===
        if should_speak(interval=10):  # Limit speech frequency
            speak("Pose detected successfully")

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks




