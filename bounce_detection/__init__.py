# Bounce Detection Module for Badminton
# 羽毛球落点检测模块

from .kinematics import KinematicsCalculator
from .candidate_generator import BounceCandidateGenerator
from .detector import BounceDetector, visualize_candidates
from .bouncenet import (
    BounceNet, 
    BounceNetLoss, 
    BounceNetPredictor,
    EVENT_LABELS, 
    LABEL_TO_EVENT,
    create_bouncenet
)
from .visual_features import VisualFeatureExtractor, PatchEncoder, TemporalPatchEncoder

__all__ = [
    # Phase 1: Rule-based detection
    'KinematicsCalculator', 
    'BounceCandidateGenerator',
    'BounceDetector',
    'visualize_candidates',
    
    # Phase 2: BounceNet classification
    'BounceNet',
    'BounceNetLoss',
    'BounceNetPredictor',
    'EVENT_LABELS',
    'LABEL_TO_EVENT',
    'create_bouncenet',
    
    # Visual features
    'VisualFeatureExtractor',
    'PatchEncoder',
    'TemporalPatchEncoder',
]
