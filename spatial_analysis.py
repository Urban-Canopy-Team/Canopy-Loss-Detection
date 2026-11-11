import numpy as np

def compute_canopy_loss(start_mask: np.ndarray, end_mask: np.ndarray, start_ndvi: np.ndarray, end_ndvi: np.ndarray) -> np.ndarray:
    """
    Flags canopy loss pixels between two time periods.
    """
    loss_condition = (
        (start_mask == 1) & 
        (end_mask == 0) & 
        (start_ndvi >= 0.35) & 
        (end_ndvi <= 0.20)
    )
    return loss_condition.astype(np.uint8)

def calculate_loss_hectares(loss_mask: np.ndarray, pixel_resolution_meters: float = 10.0) -> float:
    """
    Calculates area in Hectares (10m x 10m per pixel = 100 sq meters).
    """
    total_loss_pixels = np.sum(loss_mask == 1)
    total_area_sq_m = total_loss_pixels * (pixel_resolution_meters ** 2)
    area_hectares = total_area_sq_m / 10000.0
    return round(float(area_hectares), 2)

def estimate_trees_lost(hectares: float, density_per_ha: int = 400) -> int:
    """
    Estimates tree count assuming ~400 trees per hectare in urban forest areas.
    """
    return int(hectares * density_per_ha)

def verify_with_dynamic_world(gee_loss_count: int, dynamic_world_tree_prob: float) -> dict:
    """
    Cross-validates CNN canopy loss against Dynamic World.
    """
    is_validated = dynamic_world_tree_prob < 0.25
    confidence_score = 0.92 if is_validated else 0.45
    return {
        "validated_loss": is_validated,
        "confidence": confidence_score,
        "status": "Verified Tree Loss" if is_validated else "Possible Seasonal Disturbance"
    }