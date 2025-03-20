import cv2
import mediapipe as mp
from hand_detection import HandDetector
from palm_detection import PalmDetector
from finger_detection import FingerDetector
from gesture_recognition import GestureRecognizer

class HandLandmarkSystem:
    def __init__(self):
        # Initialize detectors
        self.hand_detector = HandDetector()
        self.palm_detector = PalmDetector()
        self.finger_detector = FingerDetector()
        self.gesture_recognizer = GestureRecognizer()

    def process_frame(self, frame):
        """Process a video frame and detect all hand landmarks."""
        # Detect hands using MediaPipe
        results = self.hand_detector.detect_hands(frame)

        # If no hands detected, return original frame
        if not results.multi_hand_landmarks:
            return frame, None

        # Get landmarks for the first detected hand
        landmarks = results.multi_hand_landmarks[0]
        landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in landmarks.landmark]

        # Find palm center and radius
        palm_center, palm_radius = self.palm_detector.find_palm_center(landmarks)

        # Estimate wrist position
        wrist = landmarks[0]

        # Detect fingertips
        fingertips = self.finger_detector.detect_fingertips(landmarks)

        # Label fingers (thumb, index, etc.)
        labeled_fingers = self.finger_detector.label_fingers(fingertips, wrist)

        # Assuming you have determined the handedness of the detected hand
        handedness = 'right'  # or 'left', based on your detection logic

        # Recognize gesture
        gesture = self.gesture_recognizer.recognize_gesture(landmarks, labeled_fingers, handedness)

        # Collect all landmarks data
        landmarks_data = {
            'palm_center': palm_center,
            'palm_radius': palm_radius,
            'wrist': wrist,
            'fingertips': fingertips,
            'labeled_fingers': labeled_fingers,
            'gesture': gesture
        }

        # Draw landmarks on the frame
        self._draw_landmarks(frame, landmarks_data)

        # Visualize finger joints
        self.visualize_finger_joints(frame, labeled_fingers)

        return frame, landmarks_data

    def _draw_landmarks(self, frame, landmarks):
        """Draw all detected landmarks on the frame."""
        # Draw palm center and circle
        cv2.circle(frame, landmarks['palm_center'], 5, (0, 0, 255), -1)
        cv2.circle(frame, landmarks['palm_center'], landmarks['palm_radius'], (0, 255, 255), 2)

        # Draw wrist point
        cv2.circle(frame, landmarks['wrist'], 8, (255, 255, 0), -1)
        cv2.putText(frame, "Wrist", landmarks['wrist'],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw skeletal lines and fingertips
        neon_colors = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (255, 0, 0)     # Red
        ]

        for i, (finger_name, tip) in enumerate(landmarks['labeled_fingers'].items()):
            # Draw line from wrist to fingertip
            cv2.line(frame, landmarks['wrist'], tip, neon_colors[i], 2)

            # Draw fingertip
            cv2.circle(frame, tip, 8, neon_colors[i], -1)
            cv2.putText(frame, finger_name, tip,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, neon_colors[i], 2)

        # Display number of fingertips only
        cv2.putText(frame, f"Fingertips: {len(landmarks['fingertips'])}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def visualize_finger_joints(self, frame, labeled_fingers):
        """Draw black dots on the finger joints."""
        for finger_name, joint_positions in labeled_fingers.items():
            if isinstance(joint_positions, list):
                for joint in joint_positions:
                    if isinstance(joint, (list, tuple)) and len(joint) == 2:
                        x, y = int(joint[0]), int(joint[1])
                        cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)  # Draw a black dot at each joint

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Change index if necessary
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize hand landmark system
    system = HandLandmarkSystem()

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process frame
        result_frame, landmarks_data = system.process_frame(frame)

        # Display result
        cv2.imshow('Hand Landmarks', result_frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()