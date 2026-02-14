"""
Configuration settings for the Gesture-to-Text system
"""

class Config:
    """Configuration class for system parameters"""
    
    # Hyperdimensional Computing Parameters
    HV_DIMENSIONS = 10000  # Dimensionality of hypervectors
    BUNDLE_THRESHOLD = 0.7  # Similarity threshold for recognition
    
    # Hand Tracking Parameters
    HAND_DETECTION_CONFIDENCE = 0.5
    HAND_TRACKING_CONFIDENCE = 0.5
    
    # Gesture Processing Parameters
    MIN_TRAJECTORY_LENGTH = 10  # Minimum points for valid gesture
    TRAJECTORY_SMOOTHING_WINDOW = 5  # Moving average window size
    
    # Camera Parameters
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # UI Parameters
    TRAJECTORY_COLOR = (0, 255, 0)  # Green
    FINGERTIP_COLOR = (0, 0, 255)   # Red
    TEXT_COLOR = (255, 255, 255)    # White
    
    # Voice Parameters
    VOICE_RATE = 150  # Words per minute
    VOICE_VOLUME = 0.9
