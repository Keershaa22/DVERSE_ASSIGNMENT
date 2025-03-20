import numpy as np
import cv2  # Ensure OpenCV is imported for visualization
from palm_detection import PalmDetector  # Ensure this is imported

# Define finger names for both hands
left_hand_fingers = {
    0: 'thumb',
    1: 'index',
    2: 'middle',
    3: 'ring',
    4: 'pinky'
}

right_hand_fingers = {
    0: 'thumb',  # Corrected naming
    1: 'index',
    2: 'middle',
    3: 'ring',
    4: 'pinky'
}

class GestureRecognizer:
    def __init__(self):
        self.palm_detector = PalmDetector()  # Initialize the palm detector

    def get_finger_names(self, handedness):
        """Return the correct finger names based on handedness."""
        if handedness == 'left':
            return left_hand_fingers
        elif handedness == 'right':
            return right_hand_fingers
        else:
            raise ValueError("Invalid handedness")

    def visualize_finger_joints(self, image, labeled_fingers):
        """Draw black dots on the finger joints."""
        for finger_name, joint_positions in labeled_fingers.items():
            # Ensure joint_positions is a list of tuples
            if isinstance(joint_positions, list):
                for joint in joint_positions:
                    if isinstance(joint, (list, tuple)) and len(joint) == 2:  # Check if joint is a tuple/list with 2 elements
                        x, y = int(joint[0]), int(joint[1])  # Assuming joint is in (x, y) format
                        cv2.circle(image, (x, y), 5, (0, 0, 0), -1)  # Draw a black dot at each joint

    def recognize_gesture(self, landmarks, labeled_fingers, handedness):
        """Recognize gestures based on landmarks and handedness."""
        finger_names = self.get_finger_names(handedness)

        # Example of using PalmDetector to find palm center
        palm_center, _ = self.palm_detector.find_palm_center(landmarks)

        # Currently, we are not recognizing any specific gestures
        return "No specific gesture recognized"