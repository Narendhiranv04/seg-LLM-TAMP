"""3D Geometric utilities for symbolic scene state resolution."""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

from llm_pipeline.region_aliases import normalize_region_name


def is_inside_xy(point: Tuple[float, float, float], world_min: np.ndarray, world_max: np.ndarray, padding: float = 0.08) -> bool:
    """Check if point is within the X-Y footprint with generous padding."""
    return (point[0] >= world_min[0] - padding and point[0] <= world_max[0] + padding and
            point[1] >= world_min[1] - padding and point[1] <= world_max[1] + padding)


def compute_dist(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def resolve_region(obj_pos: Tuple[float, float, float], region_map: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[str, str]:
    """
    Resolve region based on X-Y footprint and Z-height context.
    Handles 'Zero-Thickness' planes (boundaries) by checking proximity above the surface.
    Returns: (region_id, human_readable_description)
    """
    # Priority order: most specific regions first
    priority = [
        'box_inside_fallback',
        'box_lid_top',
        'placement_boundary',
        'cupboard_upper',
        'cupboard_lower',
        'groceries_boundary',
        'table'
    ]
    
    z = obj_pos[2]
    
    for region_name in priority:
        region_name = normalize_region_name(region_name)
        if region_name not in region_map:
            continue
            
        w_min, w_max = region_map[region_name]
        
        # 1. Check X-Y Footprint (use larger padding for boundaries)
        pad = 0.08 if 'boundary' in region_name else 0.04
        if not is_inside_xy(obj_pos, w_min, w_max, padding=pad):
            continue
            
        # 2. Check Z-Context (Proximity)
        z_min = w_min[2]
        z_max = w_max[2]
        z_thickness = z_max - z_min
        
        # If it's a plane (thickness < 1cm) or boundary marker
        if z_thickness < 0.01 or 'boundary' in region_name:
            # BROAD check: We allow object centers to be within 20cm above or BELOW the marker plane.
            # This handles models where the boundary plane might be floating slightly or 
            # where the object centroid is offset.
            if z >= z_min - 0.15 and z <= z_min + 0.20:
                if region_name == 'placement_boundary':
                    return region_name, "on table"
                if region_name == 'cupboard_lower':
                    return region_name, "inside cupboard"
                if region_name == 'cupboard_upper':
                    return region_name, "on top cupboard shelf"
                if region_name == 'box_inside_fallback':
                    return region_name, "inside the box"
                if region_name == 'groceries_boundary':
                    return region_name, "on table"
        
        # If it's a volumetric region (like 'box-top' or 'table' surface plane with height)
        else:
            if z >= z_min - 0.05 and z <= z_max + 0.10:
                if region_name == 'table':
                    return region_name, "on table"
                if region_name == 'box_lid_top':
                    return region_name, "on top of the box"

    # Default fallback
    return 'table', "on table"
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
