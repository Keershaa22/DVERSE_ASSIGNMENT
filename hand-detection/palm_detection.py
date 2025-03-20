import numpy as np

class PalmDetector:
    def __init__(self):
        pass

    def find_palm_center(self, landmarks):
        """Find palm center using MediaPipe hand landmarks."""
        # Palm landmarks are roughly landmarks 0, 1, 5, 9, 13, 17
        palm_points = [
            landmarks[0],  # Wrist
            landmarks[1],  # Thumb base
            landmarks[5],  # Index base
            landmarks[9],  # Middle base
            landmarks[13], # Ring base
            landmarks[17]  # Pinky base
        ]

        # Calculate centroid of palm points
        palm_points_array = np.array(palm_points)
        palm_center = np.mean(palm_points_array, axis=0).astype(int)

        # Calculate palm radius as the average distance from center to palm points
        distances = [np.linalg.norm(palm_center - point) for point in palm_points]
        palm_radius = int(np.mean(distances))

        return tuple(palm_center), palm_radius