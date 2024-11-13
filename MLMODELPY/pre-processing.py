import os
import re
import subprocess
import pandas as pd
import cupy as cp
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support


def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees."""
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


def extract_gps_from_exif(filepath):
    """Extract GPS coordinates from EXIF metadata using exiftool, with correct sign handling."""
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

        # Convert to decimal format, with correct sign for latitude and longitude
        decimal_lat = dms_to_decimal(*lat_parts, lat_ref)
        decimal_lon = dms_to_decimal(*lon_parts, lon_ref)

        return (decimal_lat, decimal_lon)

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None


def parallel_gps_extraction(images_dir):
    print("Starting parallel GPS extraction...")
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith('.tif')]
    with Pool(cpu_count()) as pool:
        gps_data = pool.map(extract_gps_from_exif, image_paths)
    valid_data = [(image_paths[i], gps) for i, gps in enumerate(gps_data) if gps is not None]
    print(f"Extracted GPS data for {len(valid_data)} images.")
    return valid_data


def calculate_distances_on_gpu(ground_truth, image_data):
    """Calculate distances between points using GPU acceleration, with absolute longitude values for a temporary fix."""
    try:
        # Extract and confirm coordinates, using absolute values for longitudes
        gt_lats = np.array([row['Latitude'] for row in ground_truth if not pd.isna(row['Latitude'])])
        gt_lons = np.abs(np.array([row['Longitude'] for row in ground_truth if not pd.isna(row['Longitude'])]))
        img_lats, img_lons = zip(*[gps for _, gps in image_data])
        img_lats = np.array(img_lats)
        img_lons = np.abs(np.array(img_lons))  # Use absolute values for image longitudes

        # Debug: Print coordinate samples
        print("\nGround Truth Coordinates (sample):")
        for lat, lon in zip(gt_lats[:3], gt_lons[:3]):
            print(f"Lat: {lat:.6f}, Lon (abs): {lon:.6f}")

        print("\nImage Coordinates (sample):")
        for lat, lon in zip(img_lats[:3], img_lons[:3]):
            print(f"Lat: {lat:.6f}, Lon (abs): {lon:.6f}")

        # Convert to radians
        gt_lat_arr = cp.radians(cp.array(gt_lats)).reshape(-1, 1)
        gt_lon_arr = cp.radians(cp.array(gt_lons)).reshape(-1, 1)
        img_lat_arr = cp.radians(cp.array(img_lats)).reshape(1, -1)
        img_lon_arr = cp.radians(cp.array(img_lons)).reshape(1, -1)

        # Debug: Print radian samples
        print("\nConverted Ground Truth Coordinates (Radians):")
        print(f"Lat: {gt_lat_arr[:3].flatten()}, Lon (abs): {gt_lon_arr[:3].flatten()}")

        print("\nConverted Image Coordinates (Radians):")
        print(f"Lat: {img_lat_arr[0, :3]}, Lon (abs): {img_lon_arr[0, :3]}")

        # Haversine distance calculation
        dlat = img_lat_arr - gt_lat_arr
        dlon = img_lon_arr - gt_lon_arr

        a = cp.sin(dlat / 2) ** 2 + cp.cos(gt_lat_arr) * cp.cos(img_lat_arr) * cp.sin(dlon / 2) ** 2
        a = cp.clip(a, 0, 1)  # Clip to avoid any rounding errors outside valid range
        c = 2 * cp.arcsin(cp.sqrt(a))

        # Radius of Earth in meters
        R = 6371000  # Earth radius in meters
        distances = R * c

        # Debug: Check a few calculated distances
        print("\nSample Distances (first 3x3 matrix):")
        print(cp.asnumpy(distances[:3, :3]))

        return cp.asnumpy(distances)

    except Exception as e:
        print(f"Error in GPU distance calculation: {str(e)}")
        raise

def match_ground_truth(ground_truth, image_data, distance_threshold):
    """
    Match ground truth data to closest images within a given distance threshold.
    - distance_threshold: maximum distance in meters for a ground truth point to be considered 'matched' with an image.
    """
    print("Calculating distances...")
    distances = calculate_distances_on_gpu(ground_truth, image_data)
    matches = []
    print("Finding closest matches within threshold...")

    for i in range(distances.shape[0]):
        # Get all matches within the threshold distance
        for j in range(distances.shape[1]):
            distance = distances[i, j]
            if distance <= distance_threshold:
                img_lat, img_lon = image_data[j][1]  # Image's latitude and longitude
                closest_img = image_data[j][0]  # Image file path

                # Round image coordinates to 7 decimal places for consistency
                img_lat = round(img_lat, 7)
                img_lon = round(img_lon, 7)

                # Append match information
                matches.append({
                    "GroundTruth_Lat": round(ground_truth[i]['Latitude'], 7),
                    "GroundTruth_Lon": round(ground_truth[i]['Longitude'], 7),
                    "Image_Lat": img_lat,
                    "Image_Lon": img_lon,
                    "Distance": distance,
                    "ImageFile": closest_img
                })

        # Print progress for every 10 ground truth points
        if i % 10 == 0:
            print(f"Processed {i + 1}/{len(ground_truth)} ground truth points...")

    return matches


def main():
    ground_truth_path = 'C:/Users/Jahin Catalan Mahbub/My Drive (mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/StrawberryNDVI_ChlorophyllData_Cristobal/09202024 strawberry plot, ndvi chlorophyll(09202024 strawberry ndvi) (1).csv'
    print("Loading ground truth data...")
    ground_truth_data = pd.read_csv(ground_truth_path, skiprows=4)
    ground_truth_data.columns = ['Longitude', 'Latitude', 'No', 'Red', 'NIR', 'NDVI', 'Plant_health', 'Chlorophyll']
    ground_truth = ground_truth_data.dropna(subset=['Latitude', 'Longitude']).to_dict(orient="records")

    # Directory with .tif images
    images_dir = 'M:/Workz/CPP Canvas/CS6910RA/CPP_Drone_Strawberry_F23/20240920_Strawberry_Multispectral'
    image_data = parallel_gps_extraction(images_dir)

    print("Starting GPU-accelerated ground truth matching...")
    distance_threshold = 10.0  # Adjust as necessary
    matches = match_ground_truth(ground_truth, image_data, distance_threshold)

    # Convert matches to DataFrame and save results
    matches_df = pd.DataFrame(matches)
    output_path = 'matches_output.csv'
    matches_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    freeze_support()
    main()