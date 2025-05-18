import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import numpy as np
# from calculateAngle import calculateAngle

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True,
                    min_detection_confidence=0.5, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

import math

def calculateAngle(a,b,c):
    a,b,c = np.array(a), np.array(b), np.array(c)   
    radians=np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)   

    return angle if angle <=180 else 360 - angle


def getPoseCorrection(pose, angles):
    corrections = []

    # Define same expected ranges
    pose_ranges = {
        "Mountain Pose": {'left_elbow': (160, 200), 'right_elbow': (160, 200),
                          'left_knee': (170, 200), 'right_knee': (170, 200),
                          'left_shoulder': (160, 200), 'right_shoulder': (160, 200)},
        
        "Tree Pose": {'left_knee': (40, 100), 'right_knee': (170, 200)},
        
        "Warrior II Pose": {'left_knee': (80, 110), 'right_knee': (170, 200),
                            'left_elbow': (160, 200), 'right_elbow': (160, 200)},
        
        "Triangle Pose": {'left_hip': (20, 70), 'right_hip': (160, 200),
                          'left_knee': (160, 200), 'right_knee': (160, 200)},
        
        "Chair Pose": {'left_knee': (80, 110), 'right_knee': (80, 110),
                       'left_hip': (70, 110), 'right_hip': (70, 110)},
        
        "Downward Dog Pose": {'left_hip': (90, 130), 'right_hip': (90, 130),
                              'left_shoulder': (160, 200), 'right_shoulder': (160, 200)},
        
        "Cobra Pose": {'left_elbow': (70, 110), 'right_elbow': (70, 110),
                       'left_hip': (160, 200), 'right_hip': (160, 200)},
        
        "Bridge Pose": {'left_knee': (70, 110), 'right_knee': (70, 110),
                        'left_hip': (140, 180), 'right_hip': (140, 180)},
        
        "T Pose": {'left_shoulder': (80, 100), 'right_shoulder': (80, 100),
                   'left_elbow': (160, 200), 'right_elbow': (160, 200)},
    }

    if pose not in pose_ranges:
        return ["Try adjusting to a more defined pose."]

    expected = pose_ranges[pose]

    for joint, (min_angle, max_angle) in expected.items():
        if joint in angles:
            angle = angles[joint]
            if angle < min_angle:
                corrections.append(f"{joint.replace('_', ' ').capitalize()} is too bent.")
            elif angle > max_angle:
                corrections.append(f"{joint.replace('_', ' ').capitalize()} is too extended.")

    if not corrections:
        corrections.append("Good posture!")

    return corrections


def classifyPose(landmarks):
    angles = {
        'left_elbow': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
        'right_elbow': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
        'left_shoulder': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        'right_shoulder': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]),
        'left_knee': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        'right_knee': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        'left_hip': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
        'right_hip': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
    }

    # Define expected angle ranges for each pose
    pose_ranges = {
        "Mountain Pose": {'left_elbow': (160, 200), 'right_elbow': (160, 200),
                          'left_knee': (170, 200), 'right_knee': (170, 200),
                          'left_shoulder': (160, 200), 'right_shoulder': (160, 200)},
        
        "Tree Pose": {'left_knee': (40, 100), 'right_knee': (170, 200)},
        
        "Warrior II Pose": {'left_knee': (80, 110), 'right_knee': (170, 200),
                            'left_elbow': (160, 200), 'right_elbow': (160, 200)},
        
        "Triangle Pose": {'left_hip': (20, 70), 'right_hip': (160, 200),
                          'left_knee': (160, 200), 'right_knee': (160, 200)},
        
        "Chair Pose": {'left_knee': (80, 110), 'right_knee': (80, 110),
                       'left_hip': (70, 110), 'right_hip': (70, 110)},
        
        "Downward Dog Pose": {'left_hip': (90, 130), 'right_hip': (90, 130),
                              'left_shoulder': (160, 200), 'right_shoulder': (160, 200)},
        
        "Cobra Pose": {'left_elbow': (70, 110), 'right_elbow': (70, 110),
                       'left_hip': (160, 200), 'right_hip': (160, 200)},
        
        "Bridge Pose": {'left_knee': (70, 110), 'right_knee': (70, 110),
                        'left_hip': (140, 180), 'right_hip': (140, 180)},
        
        "T Pose": {'left_shoulder': (80, 100), 'right_shoulder': (80, 100),
                   'left_elbow': (160, 200), 'right_elbow': (160, 200)},
    }

    best_match = "Unknown Pose"
    max_matches = 0

    for pose, expected in pose_ranges.items():
        match_count = 0
        for joint, (min_angle, max_angle) in expected.items():
            if min_angle <= angles[joint] <= max_angle:
                match_count += 1
        if match_count > max_matches and match_count >= len(expected) * 0.7:  # 70% match threshold
            best_match = pose
            max_matches = match_count

    return best_match, angles





   # Get corrections and angles for the pose
    corrections, angle_display = getPoseCorrection(landmarks, label) if label != 'Unknown Pose' else ([], [])
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        color = (0, 255, 0)
    
    # Write the label on the output image
    cv2.putText(output_image, label, (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Display angles on the right side of the image
    for i, angle in enumerate(angle_display[:6]):  # Show up to 6 angles
        cv2.putText(output_image, angle, (output_image.shape[1] - 200, 30 + i*30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    
    # Add corrections to the image
    for i, correction in enumerate(corrections[:3]):  # Show up to 3 corrections
        cv2.putText(output_image, correction, (10, 70 + i*30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
    
    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
    else:
        return output_image, label, corrections, angle_display