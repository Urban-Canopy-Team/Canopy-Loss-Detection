import os
import json
import ee

def init_gee(project_id: str = "urban-tree-canopy-loss"):
    """Initializes GEE using Service Account JSON in production or fallback locally."""
    ee_service_key = os.getenv("GEE_SERVICE_KEY")
    
    if ee_service_key:
        try:
            # Parse JSON string from environment variable
            key_dict = json.loads(ee_service_key)
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"],
                key_data=ee_service_key
            )
            ee.Initialize(credentials, project="urban-tree-canopy-loss")
            print("✅ Google Earth Engine initialized via Production Service Account!")
            return
        except Exception as e:
            print("⚠️ Service Account auth failed, attempting standard init:", e)

    # Local fallback
    try:
        ee.Initialize(project="urban-tree-canopy-loss")
        print("✅ Google Earth Engine successfully initialized!")
    except Exception as e:
        print("❌ GEE Initialization failed. Details:", e)
        raise e

def mask_sentinel2_clouds(image):
    """
    Sentinel-2 images contain a band called 'SCL' (Scene Classification Layer).
    This band flags every pixel as land, vegetation, water, cloud, or shadow.
    We filter out clouds (values 3, 8, 9, 10) to keep only clean pixels.
    """
    scl = image.select('SCL')
    # SCL values: 4 = Vegetation, 5 = Bare land, 6 = Water, 7 = Unclassified
    clear_pixels = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(clear_pixels)

def add_ndvi(image):
    """
    NDVI (Normalized Difference Vegetation Index) quantifies vegetation greenness.
    Formula: (NIR - Red) / (NIR + Red)
    In Sentinel-2: NIR is Band 8 ('B8') and Red is Band 4 ('B4').
    Values range from -1.0 to +1.0. High values (> 0.35) indicate dense tree canopy.
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def get_sentinel_composite(start_date: str, end_date: str, bbox: list):
    """
    Fetches, filters, masks clouds, and computes the median NDVI composite
    for a given date range and bounding box.
    """
    # Define Region of Interest (Bounding Box: [min_lon, min_lat, max_lon, max_lat])
    roi = ee.Geometry.Rectangle(bbox)

    # Load Sentinel-2 Level-2A Surface Reflectance Dataset
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) # Filter out heavy cloud scenes
        .map(mask_sentinel2_clouds)                         # Remove cloud pixels
        .map(add_ndvi)                                       # Attach NDVI channel
    )

    # Take the median pixel across all clean satellite scenes in that date range
    composite = collection.median().clip(roi)
    return composite, roi


# Test script when executed directly
if __name__ == "__main__":
    init_gee()
    
    # Bounding Box for Delhi-NCR study region
    delhi_bbox = [77.10, 28.50, 77.25, 28.65]
    
    print("Fetching Sentinel-2 baseline composite for Delhi...")
    composite, roi = get_sentinel_composite("2019-04-01", "2019-06-30", delhi_bbox)
    print("✅ Composite successfully fetched and processed!")