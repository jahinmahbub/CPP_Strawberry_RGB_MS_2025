import os
import re
import cv2
import rasterio
import subprocess
import pandas as pd
import numpy as np
from rasterio.transform import from_bounds
from PIL import Image, ImageDraw
from multiprocessing import Pool, cpu_count



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

def extract_orientation(filepath):
    """Extract orientation metadata from the image using exiftool."""
    try:
        result = subprocess.run(['exiftool', '-Orientation', filepath],
                                capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if 'Orientation' in line:
                orientation_str = line.split(':')[-1].strip()
                orientation_mapping = {
                    "Horizontal (normal)": 1,
                    "Rotate 180": 3,
                    "Rotate 90 CW": 6,
                    "Rotate 90 CCW": 8
                }
                return orientation_mapping.get(orientation_str, 1)  # Default to 1 if unknown
    except Exception as e:
        print(f"Error extracting orientation for {filepath}: {e}")
    return 1  # Default orientation


def correct_orientation(image, orientation, manual_correction=None):
    """
    Correct the image orientation based on EXIF metadata or manual correction.
    Args:
        image (PIL.Image): The image to correct.
        orientation (int): The orientation value from EXIF data.
        manual_correction (int): Degrees to manually rotate the image (optional).
    Returns:
        PIL.Image: Correctly oriented image.
    """
    if manual_correction is not None:
        # Apply manual rotation
        return image.rotate(manual_correction, expand=True)

    # Apply EXIF-based correction
    if orientation == 3:
        return image.rotate(180, expand=True)
    elif orientation == 6:
        return image.rotate(270, expand=True)
    elif orientation == 8:
        return image.rotate(90, expand=True)
    return image  # Default orientation

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
    scaling_factor = 0.55 / resolution
    new_width = int(drone_image.width / scaling_factor)
    new_height = int(drone_image.height / scaling_factor)
    return drone_image.resize((new_width, new_height))

def detect_row_orientation(image):
    """
    Detect the orientation of rows in the image using Hough Line Transform.
    Args:
        image (PIL.Image): Input image.
    Returns:
        float: Angle of the rows in degrees (relative to horizontal axis).
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

    if lines is None:
        print("No lines detected.")
        return 0  # Default to 0 degrees if no lines are found

    # Calculate angles of the detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta)
        if 80 < angle < 100:  # Only consider near-vertical lines
            angles.append(angle - 135)  # Convert to relative to horizontal axis

    # Return the average angle
    if angles:
        return np.mean(angles)
    else:
        print("No valid row lines detected.")
        return 0  # Default to 0 degrees if no valid lines are found

def enforce_horizontal_alignment(image):
    """
    Enforce horizontal alignment for rows in the image.
    Args:
        image (PIL.Image): Input image.
    Returns:
        PIL.Image: Rotated image with rows aligned horizontally.
    """
    # Detect row orientation
    row_angle = detect_row_orientation(image)
    print(f"Detected row angle: {row_angle:.2f} degrees")

    # Rotate to enforce horizontal alignment
    corrected_image = image.rotate(-row_angle, expand=True)
    return corrected_image

def detect_or_estimate_row_orientation(image, neighbors=None):
    """
    Detect or estimate the orientation of rows in the image.
    Args:
        image (PIL.Image): Input image.
        neighbors (list): List of neighbor row orientations (optional).
    Returns:
        float: Estimated or detected angle in degrees.
    """
    row_angle = detect_row_orientation(image)

    if row_angle == 0 and neighbors:
        # Use average orientation of neighbors if no valid rows are detected
        estimated_angle = np.mean([angle for angle in neighbors if angle is not None])
        print(f"No rows detected. Using estimated angle from neighbors: {estimated_angle:.2f}°")
        return estimated_angle

    print(f"Detected row angle: {row_angle:.2f}°")
    return row_angle


def blend_images(base_image, overlay_image):
    """
    Blend two images, prioritizing non-black pixels in the overlay.
    Args:
        base_image (np.array): Base image array (3D: channels x height x width).
        overlay_image (np.array): Overlay image array (3D: channels x height x width).
    Returns:
        np.array: Blended image array.
    """
    # Create a mask for non-black pixels in the overlay
    mask = np.any(overlay_image > 0, axis=0)

    # Replace only the non-black pixels in the base image
    blended = base_image.copy()
    blended[:, mask] = overlay_image[:, mask]

    return blended

def place_drone_images_with_horizontal_alignment_and_fallback(
    geotiff_image, bounds, gps_data, resolution, output_path, drone_altitude=0.45
):
    """
    Place drone images on a GeoTIFF canvas after aligning rows to horizontal and filling gaps.
    Args:
        geotiff_image (np.array): Base GeoTIFF image array.
        bounds (tuple): Bounding box of the GeoTIFF (min_lon, max_lon, min_lat, max_lat).
        gps_data (list): List of tuples containing image paths and GPS coordinates.
        resolution (float): GeoTIFF ground resolution (meters per pixel).
        output_path (str): Path to save the stitched GeoTIFF.
        drone_altitude (float): Altitude of drone images (in meters).
    """
    min_lon, max_lon, min_lat, max_lat = bounds
    width, height = geotiff_image.shape[2], geotiff_image.shape[1]
    stitched_image = np.zeros((3, height, width), dtype=np.uint8)  # RGB GeoTIFF canvas

    # Iterate through GPS data
    for file_path, (lat, lon) in gps_data:
        drone_image = Image.open(file_path)

        # Align rows to horizontal
        row_angle = detect_or_estimate_row_orientation(drone_image)
        drone_image = drone_image.rotate(-row_angle, expand=True)

        # Resize drone image to match GeoTIFF resolution
        drone_image = resize_drone_image(drone_image, drone_altitude, resolution)
        drone_array = np.array(drone_image)
        drone_height, drone_width = drone_array.shape[:2]

        # Map geocoordinates to pixel position
        x = int((lon - min_lon) / (max_lon - min_lon) * width)
        y = int((max_lat - lat) / (max_lat - min_lat) * height)

        print(f"Placing {file_path} at pixel coordinates ({x}, {y}), resized to ({drone_width}x{drone_height})")

        # Clip to GeoTIFF canvas dimensions
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(width, x + drone_width)
        y_end = min(height, y + drone_height)

        drone_x_start = max(0, -x)
        drone_y_start = max(0, -y)

        # Blend the drone image into the canvas
        drone_array = np.moveaxis(drone_array, -1, 0)  # Move channels first
        stitched_image[:, y_start:y_end, x_start:x_end] = blend_images(
            stitched_image[:, y_start:y_end, x_start:x_end],
            drone_array[:, drone_y_start:(y_end - y_start + drone_y_start),
                        drone_x_start:(x_end - x_start + drone_x_start)],
        )

    # Fill uncovered areas with a default color (salmon pink)
    uncovered = np.all(stitched_image == 0, axis=0)
    for c, color in enumerate([255, 182, 193]):  # Salmon pink
        stitched_image[c, uncovered] = color

    # Write stitched image as GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,  # RGB channels
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(*bounds, width, height),
    ) as dst:
        dst.write(stitched_image)

    print(f"Stitched GeoTIFF saved at: {output_path}")


def add_bounding_box_and_points(geotiff_path, output_path, rectangle_coords, gt_csv):
    """
    Add bounding box and plant points to the stitched GeoTIFF.
    Args:
        geotiff_path (str): Path to the input GeoTIFF.
        output_path (str): Path to save the modified GeoTIFF.
        rectangle_coords (list): List of (lat, lon) tuples for the bounding box.
        points (list): List of (lat, lon) tuples for the plant points.
    """
    with rasterio.open(geotiff_path) as src:
        image = np.moveaxis(src.read(), 0, -1)
        transform = src.transform
        bounds = src.bounds
        width, height = src.width, src.height

        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)

        # Draw bounding box
        min_lon, max_lon, min_lat, max_lat = bounds
        rectangle_pixels = [
            ((lon - min_lon) / (max_lon - min_lon) * width, (max_lat - lat) / (max_lat - min_lat) * height)
            for lat, lon in rectangle_coords
        ]
        draw.polygon(rectangle_pixels, outline="yellow", width=3)

        # Draw plant points
        # Load GT data
        gt_data = pd.read_csv(gt_csv, skiprows=4)
        gt_data.columns = [col.strip() for col in gt_data.columns]
        gt_data = gt_data.dropna(subset=["Longitude", "Latitude"])
        
        # Draw points
        for _, row in gt_data.iterrows():
            lon, lat = row["Longitude"], row["Latitude"]
            x = (lon - min_lon) / (max_lon - min_lon) * width
            y = (max_lat - lat) / (max_lat - min_lat) * height
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")

        canvas.save(output_path)
        print(f"GeoTIFF with bounding box and points saved at: {output_path}")


if __name__ == "__main__":
    # Paths and parameters
    geotiff_path = "strawberry_field_with_points.tif"
    stitched_output_path = "stitched_geo_with_points.tif"
    final_output_path = "stitched_geo_with_boxes_and_points.tif"
    drone_images_dir = "M:/Workz/CPP Canvas/CS6910RA/CPP_Drone_Strawberry_F23/20240920_Strawberry_Multispectral"

    # Bounding box and plant points
    rectangle_coords = [
        (34.043713, -117.811502),  # Top-left corner
        (34.043539, -117.811236),  # Top-right corner
        (34.042998, -117.811768),  # Bottom-right corner
        (34.043168, -117.812034)   # Bottom-left corner
    ]
    gt_csv_path = 'C:/Users/Jahin Catalan Mahbub/My Drive (mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/StrawberryNDVI_ChlorophyllData_Cristobal/09202024 strawberry plot, ndvi chlorophyll(09202024 strawberry ndvi) (1).csv'

    # Load GeoTIFF metadata
    geotiff_image, transform, bounds, dimensions = load_geotiff(geotiff_path)
    _, resolution = calculate_geotiff_height(bounds, dimensions)
    
    
    # Extract GPS data and create stitched GeoTIFF
    gps_data = parallel_gps_extraction(drone_images_dir)
    place_drone_images_with_horizontal_alignment_and_fallback(geotiff_image, bounds, gps_data, resolution, stitched_output_path)


    # Add bounding box and plant points
    add_bounding_box_and_points(stitched_output_path, final_output_path, rectangle_coords, gt_csv_path)