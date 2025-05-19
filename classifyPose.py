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
    # Standing Poses
    "Mountain Pose (Tadasana)": {
        'left_elbow': (170, 180), 'right_elbow': (170, 180),
        'left_knee': (170, 180), 'right_knee': (170, 180),
        'left_shoulder': (170, 180), 'right_shoulder': (170, 180),
        'left_hip_flexion': (160, 180), 'right_hip_flexion': (160, 180),
        'spine_forward': (170, 180),
        'neck_flexion': (160, 180)
    },
    
    # Tree Pose (Left Leg Raised)
    "Tree Pose Left (Vrksasana)": {
        'left_knee': (30, 90),           # Bent left leg (raised)
        'right_knee': (170, 180),        # Straight right leg (standing)
        'left_hip_abduction': (30, 60),  # Left hip opened outward
        'spine_forward': (160, 180)
    },
    # Tree Pose (Right Leg Raised)
    "Tree Pose Right (Vrksasana)": {
        'right_knee': (30, 90),          # Bent right leg (raised)
        'left_knee': (170, 180),         # Straight left leg (standing)
        'right_hip_abduction': (30, 60), # Right hip opened outward
        'spine_forward': (160, 180)
    },
    
    # Warrior II (Right Leg Front)
    "Warrior II Right (Virabhadrasana II)": {
        'right_knee': (85, 100),         # Bent right leg (front)
        'left_knee': (170, 180),         # Straight left leg (back)
        'right_hip_flexion': (70, 90), 
        'left_hip_flexion': (160, 180),
        'left_shoulder': (80, 100), 'right_shoulder': (80, 100)
    },
    # Warrior II (Left Leg Front)
    "Warrior II Left (Virabhadrasana II)": {
        'left_knee': (85, 100),          # Bent left leg (front)
        'right_knee': (170, 180),        # Straight right leg (back)
        'left_hip_flexion': (70, 90),
        'right_hip_flexion': (160, 180),
        'left_shoulder': (80, 100), 'right_shoulder': (80, 100)
    },
    
    # Triangle Pose (Left Leg Front)
    "Triangle Pose Left (Trikonasana)": {
        'left_knee': (170, 180),         # Straight left leg (front)
        'right_knee': (170, 180),        # Straight right leg (back)
        'left_hip_flexion': (40, 70), 
        'right_hip_flexion': (160, 180),
        'spine_forward': (20, 40)
    },
    # Triangle Pose (Right Leg Front)
    "Triangle Pose Right (Trikonasana)": {
        'right_knee': (170, 180),        # Straight right leg (front)
        'left_knee': (170, 180),         # Straight left leg (back)
        'right_hip_flexion': (40, 70),
        'left_hip_flexion': (160, 180),
        'spine_forward': (20, 40)
    },
    
    # Chair Pose (Symmetrical)
    "Chair Pose (Utkatasana)": {
        'left_knee': (80, 100), 'right_knee': (80, 100),
        'left_hip_flexion': (70, 90), 'right_hip_flexion': (70, 90),
        'spine_forward': (160, 180)
    },
    
    # Downward Dog (Symmetrical)
    "Downward Dog Pose (Adho Mukha Svanasana)": {
        'left_knee': (160, 180), 'right_knee': (160, 180),
        'left_hip_flexion': (100, 130), 'right_hip_flexion': (100, 130),
        'left_shoulder': (170, 180), 'right_shoulder': (170, 180),
        'spine_forward': (150, 180)
    },
    
    # Cobra Pose (Symmetrical)
    "Cobra Pose (Bhujangasana)": {
        'left_elbow': (80, 120), 'right_elbow': (80, 120),
        'left_hip_flexion': (160, 180), 'right_hip_flexion': (160, 180),
        'spine_backbend': (20, 40)
    },
    
    # Bridge Pose (Symmetrical)
    "Bridge Pose (Setu Bandhasana)": {
        'left_knee': (80, 100), 'right_knee': (80, 100),
        'left_hip_flexion': (120, 150), 'right_hip_flexion': (120, 150),
        'spine_backbend': (30, 60)
    },
    
    # T Pose (Symmetrical)
    "T Pose": {
        'left_shoulder': (80, 100), 'right_shoulder': (80, 100),
        'left_elbow': (170, 180), 'right_elbow': (170, 180)
    },
    
    # Warrior III (Left Leg Raised)
    "Warrior III Left (Virabhadrasana III)": {
        'left_knee': (170, 180),         # Raised left leg (straight)
        'right_knee': (170, 180),        # Standing right leg (straight)
        'spine_forward': (160, 180),
        'left_shoulder': (160, 180), 'right_shoulder': (160, 180)
    },
    # Warrior III (Right Leg Raised)
    "Warrior III Right (Virabhadrasana III)": {
        'right_knee': (170, 180),        # Raised right leg (straight)
        'left_knee': (170, 180),         # Standing left leg (straight)
        'spine_forward': (160, 180),
        'left_shoulder': (160, 180), 'right_shoulder': (160, 180)
    },
    
    # Crow Pose (Symmetrical)
    "Crow Pose (Bakasana)": {
        'left_wrist_flexion': (30, 60), 'right_wrist_flexion': (30, 60),
        'left_knee': (40, 80), 'right_knee': (40, 80),
        'left_elbow': (70, 90), 'right_elbow': (70, 90),
        'spine_forward': (120, 160)
    },
    
    # Upward Facing Dog (Symmetrical)
    "Upward Facing Dog (Urdhva Mukha Svanasana)": {
        'left_hip_flexion': (160, 180), 'right_hip_flexion': (160, 180),
        'spine_backbend': (30, 50),
        'neck_flexion': (150, 180),
        'left_ankle_dorsiflexion': (100, 130)
    },
    
    # Half Moon Pose (Left Leg Raised)
    "Half Moon Pose Left (Ardha Chandrasana)": {
        'left_knee': (170, 180),         # Raised left leg (straight)
        'right_knee': (170, 180),        # Standing right leg (straight)
        'right_hip_abduction': (20, 40), # Right hip opened
        'spine_forward': (30, 60)
    },
    # Half Moon Pose (Right Leg Raised)
    "Half Moon Pose Right (Ardha Chandrasana)": {
        'right_knee': (170, 180),        # Raised right leg (straight)
        'left_knee': (170, 180),         # Standing left leg (straight)
        'left_hip_abduction': (20, 40),  # Left hip opened
        'spine_forward': (30, 60)
    },
    
    # Eagle Pose (Left Leg Wrapped)
    "Eagle Pose Left (Garudasana)": {
        'left_knee': (30, 60),           # Left leg wrapped (raised)
        'right_knee': (80, 100),         # Right leg standing (bent)
        'left_elbow': (80, 100), 'right_elbow': (80, 100),
        'left_hip_abduction': (40, 70),
        'spine_forward': (160, 180)
    },
    # Eagle Pose (Right Leg Wrapped)
    "Eagle Pose Right (Garudasana)": {
        'right_knee': (30, 60),          # Right leg wrapped (raised)
        'left_knee': (80, 100),          # Left leg standing (bent)
        'left_elbow': (80, 100), 'right_elbow': (80, 100),
        'right_hip_abduction': (40, 70),
        'spine_forward': (160, 180)
    }
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
        # Upper Body
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
        'left_wrist_flexion': calculateAngle(  # New
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
            landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value]
        ),
        'right_wrist_flexion': calculateAngle(  # New
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value]
        ),
        'neck_flexion': calculateAngle(  # New
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value],
            landmarks[mp_pose.PoseLandmark.NOSE.value]
        ),

        # Lower Body
        'left_knee': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        'right_knee': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        'left_hip_flexion': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
        'right_hip_flexion': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]),
        'left_hip_abduction': calculateAngle(  # New
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ),
        'right_hip_abduction': calculateAngle(  # New
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ),
        'left_ankle_dorsiflexion': calculateAngle(  # New
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        ),
        'right_ankle_dorsiflexion': calculateAngle(  # New
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        ),

        # Spine
        'spine_forward': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
        'spine_backbend': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]),

        # Asymmetrical Poses
        'front_knee': calculateAngle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ),
        'standing_knee': calculateAngle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
    }

    # Define expected angle ranges for each pose
    pose_ranges = {
    # Standing Poses
    "Mountain Pose (Tadasana)": {
        'left_elbow': (170, 180), 'right_elbow': (170, 180),
        'left_knee': (170, 180), 'right_knee': (170, 180),
        'left_shoulder': (170, 180), 'right_shoulder': (170, 180),
        'left_hip_flexion': (160, 180), 'right_hip_flexion': (160, 180),
        'spine_forward': (170, 180),
        'neck_flexion': (160, 180)
    },
    
    # Tree Pose (Left Leg Raised)
    "Tree Pose Left (Vrksasana)": {
        'left_knee': (30, 90),           # Bent left leg (raised)
        'right_knee': (170, 180),        # Straight right leg (standing)
        'left_hip_abduction': (30, 60),  # Left hip opened outward
        'spine_forward': (160, 180)
    },
    # Tree Pose (Right Leg Raised)
    "Tree Pose Right (Vrksasana)": {
        'right_knee': (30, 90),          # Bent right leg (raised)
        'left_knee': (170, 180),         # Straight left leg (standing)
        'right_hip_abduction': (30, 60), # Right hip opened outward
        'spine_forward': (160, 180)
    },
    
    # Warrior II (Right Leg Front)
    "Warrior II Right (Virabhadrasana II)": {
        'right_knee': (85, 100),         # Bent right leg (front)
        'left_knee': (170, 180),         # Straight left leg (back)
        'right_hip_flexion': (70, 90), 
        'left_hip_flexion': (160, 180),
        'left_shoulder': (80, 100), 'right_shoulder': (80, 100)
    },
    # Warrior II (Left Leg Front)
    "Warrior II Left (Virabhadrasana II)": {
        'left_knee': (85, 100),          # Bent left leg (front)
        'right_knee': (170, 180),        # Straight right leg (back)
        'left_hip_flexion': (70, 90),
        'right_hip_flexion': (160, 180),
        'left_shoulder': (80, 100), 'right_shoulder': (80, 100)
    },
    
    # Triangle Pose (Left Leg Front)
    "Triangle Pose Left (Trikonasana)": {
        'left_knee': (170, 180),         # Straight left leg (front)
        'right_knee': (170, 180),        # Straight right leg (back)
        'left_hip_flexion': (40, 70), 
        'right_hip_flexion': (160, 180),
        'spine_forward': (20, 40)
    },
    # Triangle Pose (Right Leg Front)
    "Triangle Pose Right (Trikonasana)": {
        'right_knee': (170, 180),        # Straight right leg (front)
        'left_knee': (170, 180),         # Straight left leg (back)
        'right_hip_flexion': (40, 70),
        'left_hip_flexion': (160, 180),
        'spine_forward': (20, 40)
    },
    
    # Chair Pose (Symmetrical)
    "Chair Pose (Utkatasana)": {
        'left_knee': (80, 100), 'right_knee': (80, 100),
        'left_hip_flexion': (70, 90), 'right_hip_flexion': (70, 90),
        'spine_forward': (160, 180)
    },
    
    # Downward Dog (Symmetrical)
    "Downward Dog Pose (Adho Mukha Svanasana)": {
        'left_knee': (160, 180), 'right_knee': (160, 180),
        'left_hip_flexion': (100, 130), 'right_hip_flexion': (100, 130),
        'left_shoulder': (170, 180), 'right_shoulder': (170, 180),
        'spine_forward': (150, 180)
    },
    
    # Cobra Pose (Symmetrical)
    "Cobra Pose (Bhujangasana)": {
        'left_elbow': (80, 120), 'right_elbow': (80, 120),
        'left_hip_flexion': (160, 180), 'right_hip_flexion': (160, 180),
        'spine_backbend': (20, 40)
    },
    
    # Bridge Pose (Symmetrical)
    "Bridge Pose (Setu Bandhasana)": {
        'left_knee': (80, 100), 'right_knee': (80, 100),
        'left_hip_flexion': (120, 150), 'right_hip_flexion': (120, 150),
        'spine_backbend': (30, 60)
    },
    
    # T Pose (Symmetrical)
    "T Pose": {
        'left_shoulder': (80, 100), 'right_shoulder': (80, 100),
        'left_elbow': (170, 180), 'right_elbow': (170, 180)
    },
    
    # Warrior III (Left Leg Raised)
    "Warrior III Left (Virabhadrasana III)": {
        'left_knee': (170, 180),         # Raised left leg (straight)
        'right_knee': (170, 180),        # Standing right leg (straight)
        'spine_forward': (160, 180),
        'left_shoulder': (160, 180), 'right_shoulder': (160, 180)
    },
    # Warrior III (Right Leg Raised)
    "Warrior III Right (Virabhadrasana III)": {
        'right_knee': (170, 180),        # Raised right leg (straight)
        'left_knee': (170, 180),         # Standing left leg (straight)
        'spine_forward': (160, 180),
        'left_shoulder': (160, 180), 'right_shoulder': (160, 180)
    },
    
    # Crow Pose (Symmetrical)
    "Crow Pose (Bakasana)": {
        'left_wrist_flexion': (30, 60), 'right_wrist_flexion': (30, 60),
        'left_knee': (40, 80), 'right_knee': (40, 80),
        'left_elbow': (70, 90), 'right_elbow': (70, 90),
        'spine_forward': (120, 160)
    },
    
    # Upward Facing Dog (Symmetrical)
    "Upward Facing Dog (Urdhva Mukha Svanasana)": {
        'left_hip_flexion': (160, 180), 'right_hip_flexion': (160, 180),
        'spine_backbend': (30, 50),
        'neck_flexion': (150, 180),
        'left_ankle_dorsiflexion': (100, 130)
    },
    
    # Half Moon Pose (Left Leg Raised)
    "Half Moon Pose Left (Ardha Chandrasana)": {
        'left_knee': (170, 180),         # Raised left leg (straight)
        'right_knee': (170, 180),        # Standing right leg (straight)
        'right_hip_abduction': (20, 40), # Right hip opened
        'spine_forward': (30, 60)
    },
    # Half Moon Pose (Right Leg Raised)
    "Half Moon Pose Right (Ardha Chandrasana)": {
        'right_knee': (170, 180),        # Raised right leg (straight)
        'left_knee': (170, 180),         # Standing left leg (straight)
        'left_hip_abduction': (20, 40),  # Left hip opened
        'spine_forward': (30, 60)
    },
    
    # Eagle Pose (Left Leg Wrapped)
    "Eagle Pose Left (Garudasana)": {
        'left_knee': (30, 60),           # Left leg wrapped (raised)
        'right_knee': (80, 100),         # Right leg standing (bent)
        'left_elbow': (80, 100), 'right_elbow': (80, 100),
        'left_hip_abduction': (40, 70),
        'spine_forward': (160, 180)
    },
    # Eagle Pose (Right Leg Wrapped)
    "Eagle Pose Right (Garudasana)": {
        'right_knee': (30, 60),          # Right leg wrapped (raised)
        'left_knee': (80, 100),          # Left leg standing (bent)
        'left_elbow': (80, 100), 'right_elbow': (80, 100),
        'right_hip_abduction': (40, 70),
        'spine_forward': (160, 180)
    }
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