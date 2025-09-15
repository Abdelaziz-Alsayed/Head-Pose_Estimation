"""
Head Pose Estimation Package

This package provides utilities for:
- Dataset preparation and feature extraction
- Preprocessing (normalization, transformations)
- Machine learning model training and evaluation
- Pose estimation prediction on images, videos, and real-time streams
- Visualization and smoothing of predictions
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import dataset_builder
from . import preprocessing
from . import model
from . import pose_utils
from . import predictor