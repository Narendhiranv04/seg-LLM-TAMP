"""3D Geometric utilities for symbolic scene state resolution."""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

from llm_pipeline.region_geometry import is_inside_xy, resolve_region


def compute_dist(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


class GeometricReasoner:
    """Consolidated 3D geometric reasoning for state and failure validation."""

    def __init__(self):
        pass

    def is_contained_3d(self, obj_pose: Tuple[float, float, float], region_name: str, region_pose: Tuple[float, float, float]) -> bool:
        """
        Check if an object pose is 'contained' within a region using 3D proximity.
        This is a simplified version of the full volumetric check.
        """
        dist = compute_dist(obj_pose, region_pose)
        
        # Heuristics for containment based on region type
        if 'boundary' in region_name:
            # For boundaries/planes, we check X-Y proximity and Z-offset
            dx = abs(obj_pose[0] - region_pose[0])
            dy = abs(obj_pose[1] - region_pose[1])
            dz = abs(obj_pose[2] - region_pose[2])
            return dx < 0.15 and dy < 0.15 and dz < 0.20
            
        if 'box' in region_name:
            # Box containment is usually tighter
            return dist < 0.25
            
        # Default proximity check
        return dist < 0.30

    def resolve_region(self, obj_pos: Tuple[float, float, float], region_map: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[str, str]:
        """Wrap the standalone resolve_region logic."""
        return resolve_region(obj_pos, region_map)
