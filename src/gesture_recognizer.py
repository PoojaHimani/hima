"""
Pure Spatio-Temporal Hyperdimensional Computing (STHDC) Gesture Recognizer

This module implements gesture recognition using ONLY:
- Associative memory with hypervector storage
- One-shot learning (no training, epochs, or backpropagation)
- Cosine similarity for matching
- Pure mathematical operations (no deep learning)

NO CNN, NO TRAINING, NO BACKPROPAGATION, NO GRADIENT DESCENT
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from src.hypervector_encoder import HypervectorEncoder
from src.utils.config import Config

class GestureRecognizer:
    """
    Pure STHDC Gesture Recognizer with One-Shot Learning
    
    This class implements gesture recognition using associative memory
    and hyperdimensional computing without any machine learning training.
    """
    
    def __init__(self, hv_encoder: HypervectorEncoder, similarity_threshold: float = 0.7):
        """
        Initialize the STHDC gesture recognizer
        
        Args:
            hv_encoder: Hypervector encoder instance
            similarity_threshold: Minimum similarity for recognition
        """
        self.hv_encoder = hv_encoder
        self.similarity_threshold = similarity_threshold
        
        # Associative memory for storing gesture hypervectors
        # Key: gesture label (string), Value: hypervector (numpy array)
        self.associative_memory = {}
        
        # Statistics
        self.recognition_count = 0
        self.learning_count = 0
        
        print(f"🧠 STHDC Gesture Recognizer initialized")
        print(f"📊 Similarity threshold: {similarity_threshold}")
        print(f"🎯 One-shot learning enabled")
        
        # Load existing patterns if available
        self.load_patterns()
    
    def recognize(self, gesture_hv: np.ndarray) -> Optional[str]:
        """
        Recognize a gesture using associative memory lookup
        
        This function performs pure mathematical comparison without any
        machine learning inference or neural networks.
        
        Args:
            gesture_hv: Hypervector representation of gesture
            
        Returns:
            str: Recognized gesture label or None if below threshold
        """
        if gesture_hv is None or len(gesture_hv) == 0:
            return None
        
        if not self.associative_memory:
            print("⚠️ No gestures in associative memory. Use 'l' key to learn gestures first.")
            return None
        
        best_match = None
        best_similarity = -1.0
        similarities = {}
        
        # Compare input hypervector with all stored patterns
        # This is pure mathematical computation, no ML inference
        for label, stored_hv in self.associative_memory.items():
            # Compute cosine similarity between hypervectors
            similarity = self.hv_encoder.compute_cosine_similarity(gesture_hv, stored_hv)
            similarities[label] = similarity
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = label
        
        # Apply threshold for recognition
        if best_similarity >= self.similarity_threshold:
            self.recognition_count += 1
            print(f"✅ Recognized: '{best_match}' (similarity: {best_similarity:.3f})")
            return best_match
        else:
            print(f"❌ No match found (best: {best_match if best_match else 'None'} @ {best_similarity:.3f})")
            return None
    
    def learn_gesture(self, label: str, trajectory: List[Tuple[float, float, float]]) -> bool:
        """
        Learn a new gesture using one-shot learning
        
        This function stores a gesture hypervector in associative memory
        without any training, optimization, or backpropagation.
        
        Args:
            label: Label for the gesture
            trajectory: Gesture trajectory coordinates
            
        Returns:
            bool: True if learning successful
        """
        if not trajectory or len(trajectory) < 5:
            print("❌ Trajectory too short for learning")
            return False
        
        # Encode trajectory to hypervector using STHDC
        gesture_hv = self.hv_encoder.encode_trajectory(trajectory)
        
        # Store in associative memory (one-shot learning)
        self.associative_memory[label] = gesture_hv
        
        self.learning_count += 1
        print(f"🎓 Learned gesture: '{label}' ({len(trajectory)} points)")
        print(f"📝 Total gestures in memory: {len(self.associative_memory)}")
        
        # Save patterns to persistent storage
        self.save_patterns()
        
        return True
    
    def forget_gesture(self, label: str) -> bool:
        """
        Remove a gesture from associative memory
        
        Args:
            label: Label to forget
            
        Returns:
            bool: True if gesture was removed
        """
        if label in self.associative_memory:
            del self.associative_memory[label]
            print(f"🗑️ Forgot gesture: '{label}'")
            self.save_patterns()
            return True
        return False
    
    def clear_memory(self):
        """Clear all learned gestures from associative memory"""
        count = len(self.associative_memory)
        self.associative_memory.clear()
        print(f"�️ Cleared {count} gestures from memory")
        self.save_patterns()
    
    def get_all_labels(self) -> List[str]:
        """Get list of all learned gesture labels"""
        return list(self.associative_memory.keys())
    
    def get_memory_info(self) -> Dict:
        """Get information about associative memory"""
        return {
            'total_gestures': len(self.associative_memory),
            'recognition_count': self.recognition_count,
            'learning_count': self.learning_count,
            'similarity_threshold': self.similarity_threshold,
            'hv_dimensions': self.hv_encoder.get_dimensionality(),
            'labels': self.get_all_labels()
        }
    
    def set_similarity_threshold(self, threshold: float):
        """
        Update the similarity threshold for recognition
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        print(f"� Similarity threshold updated to: {self.similarity_threshold}")
    
    def save_patterns(self, filepath: str = "data/sthdc_patterns.json"):
        """
        Save learned patterns to file for persistence
        
        Args:
            filepath: Path to save patterns
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            data = {
                'patterns': {k: v.tolist() for k, v in self.associative_memory.items()},
                'metadata': {
                    'similarity_threshold': self.similarity_threshold,
                    'recognition_count': self.recognition_count,
                    'learning_count': self.learning_count,
                    'hv_dimensions': self.hv_encoder.get_dimensionality()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Saved {len(self.associative_memory)} patterns to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving patterns: {e}")
    
    def load_patterns(self, filepath: str = "data/sthdc_patterns.json"):
        """
        Load patterns from file
        
        Args:
            filepath: Path to load patterns from
        """
        try:
            if not os.path.exists(filepath):
                print(f"📂 No existing patterns file found at {filepath}")
                return
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            self.associative_memory = {
                k: np.array(v) for k, v in data.get('patterns', {}).items()
            }
            
            # Load metadata
            metadata = data.get('metadata', {})
            self.similarity_threshold = metadata.get('similarity_threshold', 0.7)
            self.recognition_count = metadata.get('recognition_count', 0)
            self.learning_count = metadata.get('learning_count', 0)
            
            print(f"📂 Loaded {len(self.associative_memory)} patterns from {filepath}")
            
        except Exception as e:
            print(f"❌ Error loading patterns: {e}")
