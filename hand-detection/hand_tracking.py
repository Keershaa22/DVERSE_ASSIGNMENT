import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def detect_hands():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image and detect hands
            results = hands.process(image_rgb)

            # If hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Draw black dots at each joint
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)  # Draw black dot at each joint

            # Display the resulting frame
            cv2.imshow('Hand Tracking', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hands()