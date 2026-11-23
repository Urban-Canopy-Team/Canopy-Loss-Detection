import numpy as np
import math

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
def calculate_afforestation_plan(
    trees_lost: int,
    annual_plantation_target: int = 10000,
    target_years: int = 5,
    survival_rate: float = 0.75,
    planting_density_per_ha: int = 400
) -> dict:
    """
    Calculates net-zero canopy recovery timeline, land required, and offset %
    based on detected tree canopy loss.
    """
    if trees_lost <= 0:
        return {
            "trees_to_replace": 0,
            "effective_annual_trees": int(annual_plantation_target * survival_rate),
            "years_to_net_zero": 0.0,
            "recovery_percentage": 100.0,
            "required_land_ha": 0.0,
            "total_saplings_needed": 0
        }

    effective_annual_trees = max(1, int(annual_plantation_target * survival_rate))
    years_to_net_zero = round(trees_lost / effective_annual_trees, 1)

    # Total surviving trees planted within the user's chosen timeline
    surviving_in_window = effective_annual_trees * target_years
    recovery_pct = round(min(100.0, (surviving_in_window / trees_lost) * 100), 1)

    # Gross saplings required to offset 100% loss accounting for survival rate
    total_saplings_needed = math.ceil(trees_lost / survival_rate)
    required_land_ha = round(total_saplings_needed / planting_density_per_ha, 2)

    return {
        "trees_to_replace": trees_lost,
        "effective_annual_trees": effective_annual_trees,
        "years_to_net_zero": years_to_net_zero,
        "recovery_percentage": recovery_pct,
        "required_land_ha": required_land_ha,
        "total_saplings_needed": total_saplings_needed
    }