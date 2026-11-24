import os
import json
import ee

def init_gee(project_id: str = "urban-tree-canopy-loss"):
    """Initializes GEE using Service Account JSON in production or fallback locally."""
    ee_service_key = os.getenv("GEE_SERVICE_KEY")
    
    if ee_service_key:
        try:
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
    Filters out clouds (values 3, 8, 9, 10) using the 'SCL' band.
    """
    scl = image.select('SCL')
    clear_pixels = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(clear_pixels)

def add_ndvi(image):
    """
    Computes NDVI = (NIR - Red) / (NIR + Red) using Sentinel-2 B8 and B4.
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def get_sentinel_composite(start_date: str, end_date: str, bbox: list):
    """
    Fetches, filters, masks clouds, and computes the median NDVI composite.
    """
    roi = ee.Geometry.Rectangle(bbox)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_sentinel2_clouds)
        .map(add_ndvi)
    )

    composite = collection.median().clip(roi)
    return composite, roi

def get_landsat_lst(start_date: str, end_date: str, bbox: list) -> ee.Image:
    """
    Landsat 8 aur 9 ke Thermal Band (ST_B10) se Land Surface Temperature (°C) nikalta hai.
    """
    roi = ee.Geometry.Rectangle(bbox)

    def mask_and_convert_lst(image):
        # 1. QA_PIXEL band se clouds (Bit 3) aur cloud shadows (Bit 4) filter karna
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        
        # 2. USGS Collection-2 official formula se raw pixel value ko Celsius (°C) me badalna
        lst_celsius = (
            image.select('ST_B10')
            .multiply(0.00341802)
            .add(149.0)
            .subtract(273.15)
            .rename('LST')
        )
        return lst_celsius.updateMask(cloud_mask)

    # 3. Landsat 8 aur Landsat 9 dono collections ko merge karna (2013 se 2026 tak ke liye)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    
    merged_landsat = l8.merge(l9).map(mask_and_convert_lst)
    
    # 4. Median composite return karna taaki extreme weather noise hat jaye
    return merged_landsat.median().clip(roi)


# Test script when executed directly
if __name__ == "__main__":
    init_gee()
    delhi_bbox = [77.10, 28.50, 77.25, 28.65]
    print("Fetching Sentinel-2 baseline composite for Delhi...")
    composite, roi = get_sentinel_composite("2019-04-01", "2019-06-30", delhi_bbox)
    print("✅ Composite successfully fetched and processed!")