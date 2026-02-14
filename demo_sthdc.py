#!/usr/bin/env python3
"""
Pure STHDC Demonstration Script

This script demonstrates the core STHDC operations without any deep learning:
- Random hypervector generation
- Spatial and temporal encoding
- Circular shift for temporal binding
- Binding and bundling operations
- Cosine similarity matching
- One-shot learning demonstration
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from hypervector_encoder import HypervectorEncoder
from gesture_recognizer import GestureRecognizer

def demonstrate_sthdc():
    """Demonstrate pure STHDC operations"""
    
    print("🧠 Pure Spatio-Temporal Hyperdimensional Computing Demo")
    print("=" * 60)
    print("🚀 NO TRAINING - NO CNN - NO BACKPROPAGATION")
    print("=" * 60)
    
    # Initialize STHDC encoder
    print("\n📊 Initializing STHDC Encoder...")
    encoder = HypervectorEncoder(dimensions=10000, seed=42)
    
    # Show memory info
    memory_info = encoder.get_memory_info()
    print(f"✅ Encoder initialized:")
    print(f"   - Dimensions: {memory_info['dimensions']}")
    print(f"   - Spatial HVs: {memory_info['spatial_hvs']}")
    print(f"   - Temporal HVs: {memory_info['temporal_hvs']}")
    print(f"   - Memory usage: {memory_info['total_memory_mb']:.2f} MB")
    
    # Initialize gesture recognizer
    print("\n🎯 Initializing Gesture Recognizer...")
    recognizer = GestureRecognizer(encoder, similarity_threshold=0.7)
    
    # Create sample trajectories for demonstration
    print("\n📝 Creating sample gesture trajectories...")
    
    # Sample trajectory for letter 'h'
    trajectory_h = []
    for i in range(20):
        trajectory_h.append((0.5, 0.2 + i * 0.02, 0))  # Vertical line
    for i in range(15):
        trajectory_h.append((0.5 + i * 0.02, 0.4 + i * 0.01, 0))  # Curve up
    for i in range(15):
        trajectory_h.append((0.8 + i * 0.01, 0.55 + i * 0.02, 0))  # Curve down
    
    # Sample trajectory for letter 'i'
    trajectory_i = []
    for i in range(3):
        trajectory_i.append((0.5, 0.2, 0))  # Dot
    for i in range(25):
        trajectory_i.append((0.5, 0.3 + i * 0.02, 0))  # Vertical line
    
    print(f"✅ Created trajectory for 'h': {len(trajectory_h)} points")
    print(f"✅ Created trajectory for 'i': {len(trajectory_i)} points")
    
    # Encode trajectories to hypervectors
    print("\n🔢 Encoding trajectories to hypervectors...")
    hv_h = encoder.encode_trajectory(trajectory_h)
    hv_i = encoder.encode_trajectory(trajectory_i)
    
    print(f"✅ Encoded 'h' trajectory: {hv_h.shape}")
    print(f"✅ Encoded 'i' trajectory: {hv_i.shape}")
    
    # Learn gestures using one-shot learning
    print("\n🎓 One-Shot Learning Demo...")
    recognizer.learn_gesture('h', trajectory_h)
    recognizer.learn_gesture('i', trajectory_i)
    
    # Test recognition
    print("\n🔍 Recognition Demo...")
    
    # Test with exact same trajectories (should recognize)
    recognized_h = recognizer.recognize(hv_h)
    recognized_i = recognizer.recognize(hv_i)
    
    print(f"🎯 Recognition Results:")
    print(f"   - 'h' trajectory recognized as: '{recognized_h}'")
    print(f"   - 'i' trajectory recognized as: '{recognized_i}'")
    
    # Test with slightly modified trajectory
    print("\n🧪 Testing with modified trajectory...")
    trajectory_h_modified = [(x + 0.01, y, z) for x, y, z in trajectory_h[:30]]  # Slightly offset
    hv_h_modified = encoder.encode_trajectory(trajectory_h_modified)
    recognized_modified = recognizer.recognize(hv_h_modified)
    print(f"   - Modified 'h' recognized as: '{recognized_modified}'")
    
    # Demonstrate similarity computation
    print("\n📊 Similarity Analysis...")
    similarity_hh = encoder.compute_cosine_similarity(hv_h, hv_h)
    similarity_hi = encoder.compute_cosine_similarity(hv_h, hv_i)
    similarity_hm = encoder.compute_cosine_similarity(hv_h, hv_h_modified)
    
    print(f"   - 'h' vs 'h' (identical): {similarity_hh:.3f}")
    print(f"   - 'h' vs 'i' (different): {similarity_hi:.3f}")
    print(f"   - 'h' vs 'modified h': {similarity_hm:.3f}")
    
    # Show final memory info
    print("\n📈 Final Memory Status...")
    memory_status = recognizer.get_memory_info()
    print(f"   - Total gestures: {memory_status['total_gestures']}")
    print(f"   - Learning events: {memory_status['learning_count']}")
    print(f"   - Recognition events: {memory_status['recognition_count']}")
    print(f"   - Stored labels: {', '.join(memory_status['labels'])}")
    
    print("\n✅ STHDC Demo Completed Successfully!")
    print("🎯 Key Takeaways:")
    print("   - No training required")
    print("   - One-shot learning works")
    print("   - Pure mathematical operations")
    print("   - Fast similarity-based recognition")

def demonstrate_mathematical_operations():
    """Demonstrate the core mathematical operations of STHDC"""
    
    print("\n🔬 Core Mathematical Operations Demo")
    print("=" * 40)
    
    encoder = HypervectorEncoder(dimensions=1000, seed=123)  # Smaller for demo
    
    # Generate test hypervectors
    hv1 = encoder._generate_bipolar_hv()
    hv2 = encoder._generate_bipolar_hv()
    hv3 = encoder._generate_bipolar_hv()
    
    print(f"📊 Generated 3 random bipolar hypervectors ({len(hv1)} dimensions)")
    print(f"   - HV1: {np.sum(hv1 == 1)} positives, {np.sum(hv1 == -1)} negatives")
    print(f"   - HV2: {np.sum(hv2 == 1)} positives, {np.sum(hv2 == -1)} negatives")
    print(f"   - HV3: {np.sum(hv3 == 1)} positives, {np.sum(hv3 == -1)} negatives")
    
    # Demonstrate binding
    print("\n🔗 Binding Operation (Multiplication)...")
    bound_hv = encoder._bind(hv1, hv2)
    print(f"   - Bound HV1 ⊗ HV2: {np.sum(bound_hv == 1)} positives, {np.sum(bound_hv == -1)} negatives")
    
    # Demonstrate bundling
    print("\n📦 Bundling Operation (Addition + Binarization)...")
    bundled_hv = encoder._bundle([hv1, hv2, hv3])
    print(f"   - Bundled (HV1 + HV2 + HV3): {np.sum(bundled_hv == 1)} positives, {np.sum(bundled_hv == -1)} negatives")
    
    # Demonstrate circular shift
    print("\n🔄 Circular Shift Operation...")
    shifted_hv = encoder._circular_shift(hv1, 100)
    print(f"   - Shifted HV1 by 100 positions")
    
    # Compute similarities
    print("\n📏 Similarity Measurements...")
    sim_12 = encoder.compute_cosine_similarity(hv1, hv2)
    sim_13 = encoder.compute_cosine_similarity(hv1, hv3)
    sim_1s = encoder.compute_cosine_similarity(hv1, shifted_hv)
    
    print(f"   - Cosine similarity HV1 vs HV2: {sim_12:.3f}")
    print(f"   - Cosine similarity HV1 vs HV3: {sim_13:.3f}")
    print(f"   - Cosine similarity HV1 vs Shifted HV1: {sim_1s:.3f}")
    
    print("\n✅ Mathematical Operations Demo Completed!")

if __name__ == "__main__":
    try:
        demonstrate_sthdc()
        demonstrate_mathematical_operations()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("💡 Run 'python main.py' to start the full gesture recognition system")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
