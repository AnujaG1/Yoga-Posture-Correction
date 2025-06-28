🧘 Yoga Posture Detection & Correction

This project uses MediaPipe and Python to detect, classify, and correct 9 different yoga poses in real-time. It's designed to help users practice yoga safely at home with instant feedback on their posture accuracy.
📌 Features

    ✅ Real-time pose detection using MediaPipe Pose

    🧠 Pose classification for 9 yoga asanas

    🧾 Feedback and correction suggestions based on joint angles

    🎥 Webcam integration for live pose analysis

    📊 Simple GUI (if you implemented one) or terminal-based feedback

🧘 Supported Poses

    Tadasana (Mountain Pose)

    Vrikshasana (Tree Pose)

    Trikonasana (Triangle Pose)

    Bhujangasana (Cobra Pose)

    Adho Mukha Svanasana (Downward Dog)

    Utkatasana (Chair Pose)

    Virabhadrasana (Warrior Pose)

    Navasana (Boat Pose)

    Setu Bandhasana (Bridge Pose)

🏗️ Project Structure

yoga-posture-detection/
│
├── main.py                   # Main script for running the detection
├── pose_classifier.py        # Contains logic to classify poses using landmarks
├── corrector.py              # Suggests posture corrections based on angles
├── utils.py                  # Helper functions for angle calculations etc.
├── requirements.txt          # List of dependencies
├── README.md                 # Project readme
└── assets/                   # Optional - Pose reference images, samples

🚀 Installation

    Clone the repo

git clone https://github.com/yourusername/yoga-posture-detection.git
cd yoga-posture-detection

Create virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

Install dependencies

    pip install -r requirements.txt

🧪 How to Run

python main.py

    Make sure your webcam is connected. The system will show your posture and give feedback.

🛠️ Technologies Used

    Python

    MediaPipe

    OpenCV

    NumPy

🤔 Future Work

    Add voice feedback

    Improve correction precision using ML models

    Add more poses and difficulty levels

    Export progress reports for users

🙋‍♀️ Contributing

Contributions are welcome! Please open issues or pull requests for new features, bug fixes, or improvements.
