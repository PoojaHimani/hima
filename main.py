#!/usr/bin/env python3
"""
Brain-Inspired Volumetric Hand-Gesture Textualization using Spatio-Temporal Hyperdimensional Computing
Final Year Project - Main Application Entry Point

PURE STHDC IMPLEMENTATION - NO TRAINING, NO CNN, NO BACKPROPAGATION
- One-shot learning with associative memory
- 10,000-dimensional bipolar hypervectors
- Circular shift for temporal binding
- Cosine similarity for recognition
"""

import sys
import os
import cv2
import numpy as np
from src.hand_tracker import HandTracker
from src.hypervector_encoder import HypervectorEncoder
from src.gesture_recognizer import GestureRecognizer
from src.text_output import TextOutput
from src.voice_output import VoiceOutput
from src.utils.config import Config

class GestureToTextApp:
    """Main application class for gesture-to-text conversion using pure STHDC"""
    
    def __init__(self):
        self.config = Config()
        self.hand_tracker = HandTracker()
        
        # Initialize STHDC encoder with 10,000 dimensions
        self.hv_encoder = HypervectorEncoder(dimensions=10000, seed=42)
        
        # Initialize gesture recognizer with one-shot learning
        self.gesture_recognizer = GestureRecognizer(
            self.hv_encoder, 
            similarity_threshold=0.7
        )
        
        self.text_output = TextOutput()
        self.voice_output = VoiceOutput()
        
        self.current_trajectory = []
        self.is_recording = False
        self.learning_mode = False
        self.pending_label = ""
        self.recognized_text = ""
        
    def run(self):
        """Main application loop"""
        print("🧠 Pure STHDC Gesture-to-Text System")
        print("=" * 60)
        print("🚀 NO TRAINING - ONE-SHOT LEARNING ONLY")
        print("=" * 60)
        print("Controls:")
        print("  - Press 'r' to start/stop recording gesture")
        print("  - Press 'l' to learn a new gesture (one-shot)")
        print("  - Press 'c' to clear current text")
        print("  - Press 'v' to toggle voice output")
        print("  - Press 'm' to show memory info")
        print("  - Press 'q' to quit")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display frame normally (no mirror effect)
            annotated_frame = frame
            
            # Track hand and get landmarks
            landmarks, annotated_frame = self.hand_tracker.track_hand(frame)
            
            if landmarks and self.is_recording:
                # Capture fingertip trajectory
                fingertip_pos = self.hand_tracker.get_fingertip_position(landmarks)
                if fingertip_pos:
                    self.current_trajectory.append(fingertip_pos)
                    # Draw trajectory on annotated frame
                    self._draw_trajectory(annotated_frame, self.current_trajectory)
            
            # Display status and recognized text
            self._draw_ui(annotated_frame)
            
            cv2.imshow('Pure STHDC Gesture-to-Text System', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._toggle_recording()
            elif key == ord('l'):
                self._start_learning()
            elif key == ord('c'):
                self._clear_text()
            elif key == ord('v'):
                self.voice_output.toggle_enabled()
                voice_status = "ON" if self.voice_output.enabled else "OFF"
                print(f"🔊 Voice output: {voice_status}")
            elif key == ord('m'):
                self._show_memory_info()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _toggle_recording(self):
        """Toggle recording state and process gesture when stopping"""
        if not self.is_recording:
            self.is_recording = True
            self.current_trajectory = []
            print("🔴 Recording started...")
        else:
            self.is_recording = False
            print("⏹️ Recording stopped, processing gesture...")
            self._process_gesture()
    
    def _start_learning(self):
        """Start learning mode for one-shot gesture learning"""
        if len(self.current_trajectory) < 5:
            print("❌ No gesture recorded. Record a gesture first with 'r' key.")
            return
        
        # Get label from user
        print("\n🎓 One-Shot Learning Mode")
        print("Enter a label for this gesture (e.g., 'a', 'b', 'hi'):")
        print("Type the label and press Enter...")
        
        # Simple input handling (in a real app, you might want a GUI)
        try:
            import tkinter as tk
            from tkinter import simpledialog
            
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            label = simpledialog.askstring(
                "Learn Gesture", 
                "Enter label for this gesture:",
                parent=root
            )
            
            root.destroy()
            
            if label and label.strip():
                self._learn_gesture(label.strip())
            else:
                print("❌ Learning cancelled - no label provided")
                
        except ImportError:
            # Fallback to console input
            label = input("Enter label: ").strip()
            if label:
                self._learn_gesture(label)
            else:
                print("❌ Learning cancelled")
    
    def _learn_gesture(self, label: str):
        """Learn a gesture using one-shot learning"""
        success = self.gesture_recognizer.learn_gesture(label, self.current_trajectory)
        
        if success:
            print(f"✅ Successfully learned gesture: '{label}'")
            print(f"📝 Total gestures in memory: {len(self.gesture_recognizer.get_all_labels())}")
            
            # Voice feedback
            self.voice_output.speak(f"Learned {label}")
            
            # Clear trajectory after learning
            self.current_trajectory = []
        else:
            print(f"❌ Failed to learn gesture: '{label}'")
    
    def _process_gesture(self):
        """Process recorded trajectory and recognize gesture"""
        if len(self.current_trajectory) < 5:  # Minimum trajectory length
            print("❌ Trajectory too short for recognition")
            return
        
        # Encode trajectory to hypervector using STHDC
        gesture_hv = self.hv_encoder.encode_trajectory(self.current_trajectory)
        
        # Recognize gesture using associative memory lookup
        recognized_char = self.gesture_recognizer.recognize(gesture_hv)
        
        if recognized_char:
            self.recognized_text += recognized_char
            print(f"✅ Recognized: '{recognized_char}'")
            print(f"📝 Current text: '{self.recognized_text}'")
            
            # Update text output
            self.text_output.update_text(self.recognized_text)
            
            # Voice output - speak full recognized text
            self.voice_output.speak(self.recognized_text)
        else:
            print("❌ Gesture not recognized")
            print("💡 Tip: Use 'l' to learn this gesture")
        
        # Clear trajectory after processing
        self.current_trajectory = []
    
    def _draw_trajectory(self, frame, trajectory):
        """Draw recorded trajectory on frame"""
        if len(trajectory) < 2:
            return
        
        # Convert 3D trajectory points to 2D for OpenCV drawing
        points_2d = []
        for x, y, z in trajectory:
            # Convert normalized coordinates to pixel coordinates
            h, w = frame.shape[:2]
            px = int(x * w)
            py = int(y * h)
            points_2d.append([px, py])
        
        points = np.array(points_2d, dtype=np.int32)
        
        # Draw thick, bright trajectory lines
        cv2.polylines(frame, [points], False, (0, 255, 255), 5)
        
        # Draw current fingertip position with larger, brighter circle
        if len(points) > 0:
            cv2.circle(frame, tuple(points[-1]), 12, (255, 255, 0), -1)
            cv2.circle(frame, tuple(points[-1]), 15, (0, 0, 255), 3)
    
    def _draw_ui(self, frame):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Status indicator
        if self.learning_mode:
            status_color = (255, 255, 0)  # Yellow for learning
            status_text = "LEARNING MODE"
        elif self.is_recording:
            status_color = (0, 255, 0)  # Green for recording
            status_text = "RECORDING"
        else:
            status_color = (0, 0, 255)  # Red for ready
            status_text = "READY"
        
        # Draw status background box
        cv2.rectangle(frame, (5, 5), (250, 40), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, status_color, 2)
        
        # Draw STHDC info background
        cv2.rectangle(frame, (5, 45), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"STHDC: {len(self.gesture_recognizer.get_all_labels())} gestures", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw recognized text background
        cv2.rectangle(frame, (5, h - 60), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"Text: {self.recognized_text}", (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw voice status background
        voice_status = "ON" if self.voice_output.enabled else "OFF"
        voice_color = (0, 255, 0) if self.voice_output.enabled else (0, 0, 255)
        cv2.rectangle(frame, (w - 150, h - 60), (w - 5, h - 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Voice: {voice_status}", (w - 145, h - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 1)
        
        # Draw instructions background
        cv2.rectangle(frame, (5, h - 100), (400, h - 65), (0, 0, 0), -1)
        cv2.putText(frame, "R:Record L:Learn C:Clear V:Voice M:Memory Q:Quit", 
                   (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _clear_text(self):
        """Clear the recognized text"""
        self.recognized_text = ""
        self.text_output.clear()
        print("🗑️ Text cleared")
    
    def _show_memory_info(self):
        """Display associative memory information"""
        info = self.gesture_recognizer.get_memory_info()
        hv_info = self.hv_encoder.get_memory_info()
        
        print("\n🧠 STHDC Memory Information")
        print("=" * 40)
        print(f"📊 Total gestures: {info['total_gestures']}")
        print(f"🎯 Recognitions: {info['recognition_count']}")
        print(f"🎓 Learning events: {info['learning_count']}")
        print(f"📏 Similarity threshold: {info['similarity_threshold']}")
        print(f"🔢 HV dimensions: {info['hv_dimensions']}")
        print(f"💾 Memory usage: {hv_info['total_memory_mb']:.2f} MB")
        print(f"📝 Stored labels: {', '.join(info['labels']) if info['labels'] else 'None'}")
        print("=" * 40)

def main():
    """Main entry point"""
    try:
        app = GestureToTextApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
