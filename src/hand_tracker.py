"""
Hand tracking module using MediaPipe for 3D landmark detection
"""

import cv2
import mediapipe as mp
import numpy as np
from src.utils.config import Config

class HandTracker:
    """Hand tracking class using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=Config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.HAND_TRACKING_CONFIDENCE
        )
        
        # Fingertip landmarks (MediaPipe indices)
        self.fingertip_indices = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        # Use index finger as primary writing finger
        self.primary_finger = 'index'
    
    def track_hand(self, frame):
        """
        Track hand in given frame and return landmarks
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (landmarks, annotated_frame)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        # Create annotated frame (no mirror effect)
        annotated_frame = frame.copy()
        
        landmarks = None
        if results.multi_hand_landmarks:
            # Get first hand's landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks
            
            # Draw hand landmarks on frame
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Highlight primary fingertip
            fingertip_idx = self.fingertip_indices[self.primary_finger]
            fingertip = hand_landmarks.landmark[fingertip_idx]
            h, w = annotated_frame.shape[:2]
            cx, cy = int(fingertip.x * w), int(fingertip.y * h)
            
            # Draw circle around primary fingertip
            cv2.circle(annotated_frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.circle(annotated_frame, (cx, cy), 12, (0, 0, 0), 2)
        
        return landmarks, annotated_frame
    
    def get_fingertip_position(self, landmarks):
        """
        Get the (x, y, z) position of the primary fingertip
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            tuple: (x, y) coordinates in pixel space
        """
        if not landmarks:
            return None
        
        fingertip_idx = self.fingertip_indices[self.primary_finger]
        fingertip = landmarks.landmark[fingertip_idx]
        
        # Convert normalized coordinates to pixel coordinates
        x = fingertip.x
        y = fingertip.y
        z = fingertip.z  # Depth information
        
        return (x, y, z)
    
    def get_all_fingertip_positions(self, landmarks):
        """
        Get positions of all fingertips
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            dict: Dictionary of fingertip positions
        """
        if not landmarks:
            return {}
        
        positions = {}
        for finger_name, idx in self.fingertip_indices.items():
            fingertip = landmarks.landmark[idx]
            positions[finger_name] = (fingertip.x, fingertip.y, fingertip.z)
        
        return positions
    
    def is_hand_present(self, landmarks):
        """
        Check if hand is detected
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            bool: True if hand is present
        """
        return landmarks is not None
    
    def get_hand_bounding_box(self, landmarks):
        """
        Get bounding box of the hand
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            tuple: (x_min, y_min, x_max, y_max) in normalized coordinates
        """
        if not landmarks:
            return None
        
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def normalize_trajectory(self, trajectory):
        """
        Normalize trajectory coordinates to [0, 1] range
        
        Args:
            trajectory: List of (x, y, z) coordinates
            
        Returns:
            list: Normalized trajectory
        """
        if not trajectory:
            return []
        
        trajectory = np.array(trajectory)
        
        # Normalize to [0, 1] range
        min_vals = np.min(trajectory, axis=0)
        max_vals = np.max(trajectory, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (trajectory - min_vals) / range_vals
        
        return normalized.tolist()
    
    def smooth_trajectory(self, trajectory, window_size=5):
        """
        Apply moving average smoothing to trajectory
        
        Args:
            trajectory: List of coordinates
            window_size: Size of moving average window
            
        Returns:
            list: Smoothed trajectory
        """
        if len(trajectory) < window_size:
            return trajectory
        
        trajectory = np.array(trajectory)
        smoothed = []
        
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            window = trajectory[start_idx:end_idx]
            smoothed.append(np.mean(window, axis=0))
        
        return smoothed.tolist()
