import numpy as np

def is_inside_xy(point, world_min, world_max, padding=0.08):
    """Check if point is within the X-Y footprint with generous padding."""
    return (point[0] >= world_min[0] - padding and point[0] <= world_max[0] + padding and
            point[1] >= world_min[1] - padding and point[1] <= world_max[1] + padding)

def resolve_region(obj_pos, region_map):
    """
    Resolve region based on X-Y footprint and Z-height context.
    Handles 'Zero-Thickness' planes (boundaries) by checking proximity above the surface.
    """
    # Priority order: most specific regions first
    priority = [
        'box-inside',
        'box-top',
        'placement_boundary',
        'cupboard_boundary_top',
        'cupboard_boundary',
        'groceries_boundary',
        'table'
    ]
    
    z = obj_pos[2]
    
    for region_name in priority:
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
        
        # If it's a plane (thickness < 1cm)
        if z_thickness < 0.01:
            # Broaden check: mug3 z=1.27 but boundary z=1.35. 
            # This suggests the boundary marker might be ABOVE the object center.
            # We'll allow object centers to be within 20cm above or BELOW the marker plane.
            if z >= z_min - 0.15 and z <= z_min + 0.20:
                if region_name == 'placement_boundary':
                    return region_name, "on table"
                if region_name == 'cupboard_boundary':
                    return region_name, "inside cupboard"
                if region_name == 'cupboard_boundary_top':
                    return region_name, "on top cupboard shelf"
                if region_name == 'box-inside':
                    return region_name, "inside the box"
                if region_name == 'groceries_boundary':
                    return region_name, "on table"
        
        # If it's a volume
        else:
            if z >= z_min - 0.05 and z <= z_max + 0.10:
                if region_name == 'table':
                    return region_name, "on table"
                if region_name == 'box-top':
                    return region_name, "on top of the box"

    # Default fallback
    return 'table', "on table"
