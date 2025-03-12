import rasterio
from PIL import Image, ImageDraw
import os
import re
import subprocess
from multiprocessing import Pool, cpu_count
import numpy as np


def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees."""
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


def extract_gps_from_exif(filepath):
    """Extract GPS coordinates from EXIF metadata using exiftool."""
    try:
        result = subprocess.run(['exiftool', '-GPS*', filepath],
                                capture_output=True, text=True)
        gps_data = result.stdout.strip().split('\n')
        lat_data, lat_ref, lon_data, lon_ref = None, None, None, None
        for line in gps_data:
            if 'GPS Latitude' in line and 'Ref' not in line:
                lat_data = line.split(':')[1].strip()
            elif 'GPS Latitude Ref' in line:
                lat_ref = line.split(':')[1].strip()
            elif 'GPS Longitude' in line and 'Ref' not in line:
                lon_data = line.split(':')[1].strip()
            elif 'GPS Longitude Ref' in line:
                lon_ref = line.split(':')[1].strip()

        if not all([lat_data, lat_ref, lon_data, lon_ref]):
            print(f"No GPS data found in {filepath}")
            return None

        # Parse DMS format
        def parse_dms(dms_str):
            pattern = r'(\d+) deg (\d+)\' ([\d.]+)"'
            match = re.match(pattern, dms_str)
            if match:
                return [float(x) for x in match.groups()]
            return None

        lat_parts = parse_dms(lat_data)
        lon_parts = parse_dms(lon_data)
        if not lat_parts or not lon_parts:
            return None

        # Convert to decimal format with correct sign
        decimal_lat = dms_to_decimal(*lat_parts, lat_ref)
        decimal_lon = - dms_to_decimal(*lon_parts, lon_ref)

        # Debug: Print extracted values for verification
        print(f"File: {filepath}")
        print(f"Extracted Latitude: {decimal_lat}, Longitude: {decimal_lon}")

        return (decimal_lat, decimal_lon)

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None


def parallel_gps_extraction(images_dir):
    """Extract GPS data for all images in a directory using multiprocessing."""
    print("Starting parallel GPS extraction...")
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    with Pool(cpu_count()) as pool:
        gps_data = pool.map(extract_gps_from_exif, image_paths)
    valid_data = [(image_paths[i], gps) for i, gps in enumerate(gps_data) if gps is not None]
    print(f"Extracted GPS data for {len(valid_data)} images.")
    return valid_data


def load_geotiff(geotiff_path):
    """
    Load the GeoTIFF file and extract geospatial metadata.
    Args:
        geotiff_path (str): Path to the GeoTIFF file.
    Returns:
        tuple: (image array, transform, bounds, dimensions)
    """
    with rasterio.open(geotiff_path) as src:
        image_array = src.read()
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height
    return image_array, transform, bounds, (width, height)


def calculate_geotiff_height(bounds, dimensions):
    """
    Calculate the ground height (in meters) of the GeoTIFF based on its geographic bounds.
    Args:
        bounds (tuple): (min_lon, max_lon, min_lat, max_lat) of the GeoTIFF.
        dimensions (tuple): (width, height) in pixels of the GeoTIFF.
    Returns:
        float: Height of the GeoTIFF in meters.
        float: Ground resolution (meters per pixel).
    """
    min_lat, max_lat = bounds[2], bounds[3]
    height_pixels = dimensions[1]

    # Calculate latitude difference in meters (1 degree latitude = ~111.32 km)
    lat_diff_meters = abs(max_lat - min_lat) * 111320

    # Calculate resolution (meters per pixel)
    resolution = lat_diff_meters / height_pixels

    return lat_diff_meters, resolution


def resize_drone_image(drone_image, drone_altitude, resolution):
    """
    Resize a drone image based on its altitude and the GeoTIFF resolution.
    Args:
        drone_image (PIL.Image): The drone image to resize.
        drone_altitude (float): The altitude at which the drone image was taken (in meters).
        resolution (float): Ground resolution of the GeoTIFF (meters per pixel).
    Returns:
        PIL.Image: Resized drone image.
    """
    scaling_factor = 0.45 / resolution
    new_width = int(drone_image.width / scaling_factor)
    new_height = int(drone_image.height / scaling_factor)
    return drone_image.resize((new_width, new_height))


def place_drone_images_with_gps(geotiff_image, transform, bounds, gps_data, resolution, output_path, rectangle_coords, blank_color=(255, 182, 193)):
    """
    Place drone images on the GeoTIFF canvas based on GPS coordinates, resizing them to match the GeoTIFF scale.
    Args:
        geotiff_image (np.array): Base GeoTIFF image array.
        transform (Affine): GeoTIFF transform object for coordinate mapping.
        bounds (tuple): Bounding box of the GeoTIFF (min_lon, max_lon, min_lat, max_lat).
        gps_data (list): List of tuples containing image paths and GPS coordinates.
        resolution (float): GeoTIFF ground resolution (meters per pixel).
        output_path (str): Path to save the output PNG.
        rectangle_coords (list): Hardcoded four corners of the rectangle in (lat, lon) format.
        blank_color (tuple): RGB color for blank areas (default: salmon pink).
    """
    min_lon, max_lon, min_lat, max_lat = bounds
    width, height = geotiff_image.shape[2], geotiff_image.shape[1]

    # Create a canvas with the GeoTIFF as the base
    canvas = Image.fromarray(np.moveaxis(geotiff_image[:3], 0, -1))  # Use the GeoTIFF image as the base canvas
    draw = ImageDraw.Draw(canvas)

    # Draw the bounding box rectangle on the canvas
    rectangle_pixels = [
        ((lon - min_lon) / (max_lon - min_lon) * width, (max_lat - lat) / (max_lat - min_lat) * height)
        for lat, lon in rectangle_coords
    ]
    draw.polygon(rectangle_pixels, outline="yellow", width=3)

    # Iterate through GPS data
    for file_path, (lat, lon) in gps_data:
        drone_image = Image.open(file_path)

        # Resize drone image to match GeoTIFF resolution
        drone_image = resize_drone_image(drone_image, 10, resolution)  # Drone altitude = 10m

        drone_width, drone_height = drone_image.size

        # Map geocoordinates to pixel position
        x = int((lon - min_lon) / (max_lon - min_lon) * width)
        y = int((max_lat - lat) / (max_lat - min_lat) * height)

        print(f"Placing {file_path} at pixel coordinates ({x}, {y}), resized to ({drone_width}x{drone_height})")

        # Paste the drone image onto the canvas
        canvas.paste(drone_image, (x, y, x + drone_width, y + drone_height))

    # Save the final image as PNG
    canvas.save(output_path)
    print(f"Stitched map saved at: {output_path}")


if __name__ == "__main__":
    # Path to the GeoTIFF file
    geotiff_path = "strawberry_field_with_points.tif"
    geotiff_image, transform, bounds, dimensions = load_geotiff(geotiff_path)

    # Hardcoded four corners of the bounding box
    rectangle_coords = [
        (34.043713, -117.811502),  # Top-left corner
        (34.043539, -117.811236),  # Top-right corner
        (34.042998, -117.811768),  # Bottom-right corner
        (34.043168, -117.812034)   # Bottom-left corner
    ]

    # Calculate GeoTIFF height and resolution
    geotiff_height, geotiff_resolution = calculate_geotiff_height(bounds, dimensions)
    print(f"GeoTIFF Height: {geotiff_height:.2f} meters, Resolution: {geotiff_resolution:.2f} meters per pixel")

    # Directory containing drone images
    drone_images_dir = "M:/Workz/CPP Canvas/CS6910RA/CPP_Drone_Strawberry_F23/20240920_Strawberry_Multispectral"

    # Extract GPS data from drone images
    gps_data = parallel_gps_extraction(drone_images_dir)

    # Output PNG path
    output_path = "stitched_drone_map_with_points.png"

    # Place drone images and save the result
    place_drone_images_with_gps(geotiff_image, transform, bounds, gps_data, geotiff_resolution, output_path, rectangle_coords)