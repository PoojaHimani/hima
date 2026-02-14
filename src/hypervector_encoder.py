"""
Pure Spatio-Temporal Hyperdimensional Computing (STHDC) Encoder

This module implements the core STHDC operations without any training:
- Random seed hypervectors for spatial encoding
- 10,000-dimensional bipolar hypervectors
- Circular shift for temporal encoding
- Binding (multiplication/XOR)
- Bundling (superposition)
- No training, CNN, backpropagation, or gradient descent
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from src.utils.config import Config

class HypervectorEncoder:
    """
    Pure Spatio-Temporal Hyperdimensional Computing Encoder
    
    This class implements STHDC without any machine learning training.
    All operations are deterministic mathematical transformations.
    """
    
    def __init__(self, dimensions: int = 10000, seed: int = None):
        """
        Initialize the STHDC encoder with random seed hypervectors
        
        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Generate random seed hypervectors for spatial encoding
        self.position_hvs = {}  # Spatial position hypervectors
        self.time_hvs = {}      # Temporal step hypervectors
        
        # Initialize the hypervector space
        self._initialize_seed_hypervectors()
        
        print(f"🧠 STHDC Encoder initialized: {dimensions} dimensions")
        print(f"🎲 Random seed: {seed if seed else 'None'}")
    
    def _initialize_seed_hypervectors(self):
        """
        Generate random seed hypervectors for spatial and temporal encoding.
        These are one-time generated vectors used for all encoding operations.
        """
        # Create spatial hypervectors for a grid of possible positions
        # Using 50x50 grid for normalized coordinates [0,1]
        grid_resolution = 50
        
        print(f"📍 Generating {grid_resolution}x{grid_resolution} spatial hypervectors...")
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                key = (i, j)
                # Generate random bipolar hypervector (+1/-1)
                self.position_hvs[key] = self._generate_bipolar_hv()
        
        # Create temporal hypervectors for time steps
        # Support up to 500 time steps for gesture sequences
        max_time_steps = 500
        
        print(f"⏰ Generating {max_time_steps} temporal hypervectors...")
        
        for t in range(max_time_steps):
            self.time_hvs[t] = self._generate_bipolar_hv()
        
        print("✅ Seed hypervectors generated successfully")
    
    def _generate_bipolar_hv(self) -> np.ndarray:
        """
        Generate a random bipolar hypervector (+1/-1)
        
        Returns:
            np.ndarray: Random bipolar hypervector
        """
        return np.random.choice([-1, 1], size=self.dimensions)
    
    def _get_spatial_hv(self, x: float, y: float) -> np.ndarray:
        """
        Get spatial hypervector for normalized coordinates (x, y)
        
        Args:
            x, y: Normalized coordinates in range [0, 1]
            
        Returns:
            np.ndarray: Spatial hypervector
        """
        # Convert normalized coordinates to grid indices
        grid_resolution = 50
        grid_x = int(x * (grid_resolution - 1))
        grid_y = int(y * (grid_resolution - 1))
        
        # Clamp to valid range
        grid_x = max(0, min(grid_resolution - 1, grid_x))
        grid_y = max(0, min(grid_resolution - 1, grid_y))
        
        key = (grid_x, grid_y)
        
        # Return the pre-generated spatial hypervector
        return self.position_hvs[key]
    
    def _get_temporal_hv(self, time_step: int) -> np.ndarray:
        """
        Get temporal hypervector for a given time step
        
        Args:
            time_step: Time step index in sequence
            
        Returns:
            np.ndarray: Temporal hypervector
        """
        # Return the pre-generated temporal hypervector
        return self.time_hvs[time_step]
    
    def _circular_shift(self, hv: np.ndarray, shift_amount: int) -> np.ndarray:
        """
        Apply circular shift for temporal binding
        
        This operation preserves the hypervector's properties while encoding
        temporal information through position-based shifting.
        
        Args:
            hv: Input hypervector
            shift_amount: Number of positions to shift
            
        Returns:
            np.ndarray: Temporally shifted hypervector
        """
        return np.roll(hv, shift_amount % self.dimensions)
    
    def _bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors using multiplication (XOR for bipolar vectors)
        
        Binding creates a new hypervector that represents the combination
        of two input hypervectors while preserving their individual information.
        
        Args:
            hv1, hv2: Input hypervectors
            
        Returns:
            np.ndarray: Bound hypervector
        """
        # Element-wise multiplication for bipolar vectors
        return hv1 * hv2
    
    def _bundle(self, hypervectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple hypervectors using superposition (addition)
        
        Bundling combines multiple pieces of information into a single
        hypervector through vector addition and binarization.
        
        Args:
            hypervectors: List of hypervectors to bundle
            
        Returns:
            np.ndarray: Bundled hypervector
        """
        if not hypervectors:
            return np.zeros(self.dimensions)
        
        # Vector addition (superposition)
        bundled = np.sum(hypervectors, axis=0)
        
        # Binarize the result back to bipolar
        # Positive values become +1, negative values become -1
        return np.where(bundled >= 0, 1, -1)
    
    def encode_trajectory(self, trajectory: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Encode a complete trajectory into a single hypervector using STHDC
        
        This is the core encoding function that transforms a spatio-temporal
        trajectory into a high-dimensional representation.
        
        Args:
            trajectory: List of (x, y, z) coordinates
            
        Returns:
            np.ndarray: Trajectory hypervector
        """
        if not trajectory:
            return np.zeros(self.dimensions)
        
        # Encode each point in the trajectory with spatio-temporal binding
        point_hvs = []
        
        for t, (x, y, z) in enumerate(trajectory):
            # Get spatial hypervector for current position
            spatial_hv = self._get_spatial_hv(x, y)
            
            # Get temporal hypervector for current time step
            temporal_hv = self._get_temporal_hv(t)
            
            # Apply temporal binding using circular shift
            # The shift amount is proportional to the time step
            shifted_temporal_hv = self._circular_shift(temporal_hv, t * 10)
            
            # Bind spatial and temporal information
            bound_hv = self._bind(spatial_hv, shifted_temporal_hv)
            
            point_hvs.append(bound_hv)
        
        # Bundle all points to create the complete trajectory hypervector
        trajectory_hv = self._bundle(point_hvs)
        
        return trajectory_hv
    
    def encode_stroke(self, stroke: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Encode a single stroke (continuous gesture segment)
        
        Args:
            stroke: List of coordinates for a single stroke
            
        Returns:
            np.ndarray: Stroke hypervector
        """
        return self.encode_trajectory(stroke)
    
    def bundle_strokes(self, stroke_hvs: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple strokes to form a complete gesture/word
        
        Args:
            stroke_hvs: List of stroke hypervectors
            
        Returns:
            np.ndarray: Complete gesture hypervector
        """
        return self._bundle(stroke_hvs)
    
    def compute_cosine_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors
        
        Args:
            hv1, hv2: Input hypervectors
            
        Returns:
            float: Cosine similarity in range [-1, 1]
        """
        # Compute dot product
        dot_product = np.dot(hv1, hv2)
        
        # Compute magnitudes
        mag1 = np.linalg.norm(hv1)
        mag2 = np.linalg.norm(hv2)
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (mag1 * mag2)
        
        return similarity
    
    def compute_hamming_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """
        Compute Hamming similarity for bipolar vectors
        
        Args:
            hv1, hv2: Input hypervectors
            
        Returns:
            float: Hamming similarity in range [0, 1]
        """
        if len(hv1) != len(hv2):
            return 0.0
        
        # Count matching positions
        matches = np.sum(hv1 == hv2)
        total = len(hv1)
        
        return matches / total
    
    def get_dimensionality(self) -> int:
        """Get the dimensionality of hypervectors"""
        return self.dimensions
    
    def get_memory_info(self) -> dict:
        """Get information about memory usage"""
        spatial_memory = len(self.position_hvs) * self.dimensions * 1  # 1 byte per bipolar value
        temporal_memory = len(self.time_hvs) * self.dimensions * 1
        
        return {
            'dimensions': self.dimensions,
            'spatial_hvs': len(self.position_hvs),
            'temporal_hvs': len(self.time_hvs),
            'spatial_memory_mb': spatial_memory / (1024 * 1024),
            'temporal_memory_mb': temporal_memory / (1024 * 1024),
            'total_memory_mb': (spatial_memory + temporal_memory) / (1024 * 1024)
        }
    
    def reset_seed(self, seed: int = None):
        """
        Reset the encoder with a new random seed
        
        Args:
            seed: New random seed
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Clear existing hypervectors
        self.position_hvs.clear()
        self.time_hvs.clear()
        
        # Regenerate seed hypervectors
        self._initialize_seed_hypervectors()
        
        print(f"🔄 STHDC Encoder reset with new seed: {seed}")
