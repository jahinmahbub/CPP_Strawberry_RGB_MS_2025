import os
import re
import subprocess
import pandas as pd
import cupy as cp
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support


def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS (degrees, minutes, seconds) to decimal format"""
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


def extract_gps_from_exif(filepath):
    """Extract GPS coordinates from EXIF metadata using exiftool"""
    try:
        result = subprocess.run(['exiftool', '-GPS*', filepath],
                                capture_output=True, text=True)

        # Parse the output
        gps_data = result.stdout.strip().split('\n')
        lat_data = None
        lat_ref = None
        lon_data = None
        lon_ref = None

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
            return None

        # Extract degrees, minutes, seconds using regex
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

        # Convert to decimal format
        decimal_lat = dms_to_decimal(*lat_parts, lat_ref)
        decimal_lon = dms_to_decimal(*lon_parts, lon_ref)

        return decimal_lon, decimal_lat

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None


def parallel_gps_extraction(images_dir):
    """Extract GPS data from images in parallel"""
    print("Starting parallel GPS extraction...")
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                   if f.lower().endswith('.tif')]

    # Create a pool with the number of available CPU cores
    with Pool(cpu_count()) as pool:
        gps_data = pool.map(extract_gps_from_exif, image_paths)

    # Filter out images without GPS data
    valid_data = [(image_paths[i], gps) for i, gps in enumerate(gps_data)
                  if gps is not None]
    print(f"Extracted GPS data for {len(valid_data)} images.")
    return valid_data


def calculate_distances_on_gpu(ground_truth, image_data):
    """Calculate distances between points using GPU acceleration"""
    try:
        gt_lats = np.array([row['Latitude'] for row in ground_truth])
        gt_lons = np.array([row['Longitude'] for row in ground_truth])
        img_lons, img_lats = zip(*[gps for _, gps in image_data])
        img_lats = np.array(img_lats)
        img_lons = np.array(img_lons)

        # Transfer data to GPU
        gt_lat_arr = cp.array(gt_lats)
        gt_lon_arr = cp.array(gt_lons)
        img_lat_arr = cp.array(img_lats)
        img_lon_arr = cp.array(img_lons)

        # Reshape arrays for broadcasting
        gt_lat_arr = gt_lat_arr.reshape(-1, 1)
        gt_lon_arr = gt_lon_arr.reshape(-1, 1)

        # Calculate Haversine distance components
        dlat = cp.radians(gt_lat_arr - img_lat_arr)
        dlon = cp.radians(gt_lon_arr - img_lon_arr)

        lat1_rad = cp.radians(gt_lat_arr)
        lat2_rad = cp.radians(img_lat_arr)

        # Haversine formula
        a = cp.sin(dlat / 2.0) ** 2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon / 2.0) ** 2
        c = 2 * cp.arcsin(cp.sqrt(a))

        # Earth radius in meters
        R = 6371000
        distances = R * c

        return cp.asnumpy(distances)

    except Exception as e:
        print(f"Error in GPU calculation: {str(e)}")
        raise


def match_ground_truth(ground_truth, image_data):
    """Match ground truth data to closest images"""
    print("Calculating distances...")
    distances = calculate_distances_on_gpu(ground_truth, image_data)
    matches = []

    print("Finding closest matches...")
    for i in range(distances.shape[0]):
        min_idx = int(np.argmin(distances[i, :]))  # Use numpy instead of cupy for argmin
        closest_img = image_data[min_idx][0]
        closest_distance = distances[i, min_idx]
        matches.append({
            "GroundTruth_Lat": ground_truth[i]['Latitude'],
            "GroundTruth_Lon": ground_truth[i]['Longitude'],
            "ImageFile": closest_img,
            "Distance": closest_distance
        })
        if i % 10 == 0:  # Print progress every 10 matches
            print(f"Processed {i + 1}/{len(ground_truth)} matches...")

    return matches


def main():
    # Load ground truth data
    ground_truth_path = ('C:/Users/Jahin Catalan Mahbub/My Drive '
                         '(mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/'
                         'StrawberryNDVI_ChlorophyllData_Cristobal/'
                         '09202024 strawberry plot, ndvi chlorophyll'
                         '(09202024 strawberry ndvi) (1).csv')

    print("Loading ground truth data...")
    ground_truth_data = pd.read_csv(ground_truth_path, skiprows=4)
    ground_truth_data.columns = ['Longitude', 'Latitude', 'No', 'Red', 'NIR',
                                 'NDVI', 'Plant_health', 'Chlorophyll']
    ground_truth = ground_truth_data.to_dict(orient="records")

    # Directory with .tif images
    images_dir = ('M:/Workz/CPP Canvas/CS6910RA/CPP_Drone_Strawberry_F23/'
                  '20240920_Strawberry_Multispectral')

    # Extract GPS data from images in parallel
    image_data = parallel_gps_extraction(images_dir)

    # Run matching with GPU-accelerated distance calculations
    print("Starting GPU-accelerated ground truth matching...")
    matches = match_ground_truth(ground_truth, image_data)

    # Convert matches to DataFrame and save results
    matches_df = pd.DataFrame(matches)
    print("\nFinal matches summary:")
    print(f"Total matches found: {len(matches_df)}")
    print("\nSample of matches:")
    print(matches_df.head())

    # Save the results
    output_path = 'matches_output.csv'
    matches_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    freeze_support()  # Required for Windows
    main()