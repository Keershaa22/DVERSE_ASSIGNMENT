import cv2
import numpy as np
import math

class FingerDetector:
    def __init__(self):
        pass

    def detect_fingertips(self, landmarks):
        """Detect fingertips using MediaPipe hand landmarks."""
        # Fingertip landmarks are 4, 8, 12, 16, 20
        fingertips = [
            landmarks[4],  # Thumb tip
            landmarks[8],  # Index tip
            landmarks[12], # Middle tip
            landmarks[16], # Ring tip
            landmarks[20]  # Pinky tip
        ]
        return fingertips

    def label_fingers(self, fingertips, wrist):
        """Label fingers based on their relative positions to the wrist."""
        if len(fingertips) < 1:
            return {}

        # Calculate the angle of each fingertip relative to the wrist
        angles = []
        for tip in fingertips:
            dx = tip[0] - wrist[0]
            dy = tip[1] - wrist[1]
            angle = math.degrees(math.atan2(dy, dx))
            angles.append((tip, angle))

        # Sort fingertips by angle (from left to right)
        angles.sort(key=lambda x: x[1])

        # Label fingers based on their order
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        labeled_fingers = {}
        for i, (tip, _) in enumerate(angles):
            if i < len(finger_names):
                labeled_fingers[finger_names[i]] = tip

        return labeled_fingers