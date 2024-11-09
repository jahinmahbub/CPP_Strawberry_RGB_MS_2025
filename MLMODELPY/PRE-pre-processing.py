import os
import re
import subprocess
import pandas as pd
from geopy.distance import geodesic  # Calculate distance between GPS points


# Convert DMS (degrees, minutes, seconds) to decimal format
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


# Extract GPS coordinates from EXIF metadata using exiftool
def extract_gps_from_exif(filepath):
    print(f"Extracting GPS data from {filepath}")
    try:
        result = subprocess.run(['exiftool', '-GPS*', filepath], capture_output=True, text=True)

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
            print("GPS data is incomplete.")
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
            print("GPS DMS parsing failed.")
            return None

        # Convert to decimal format
        decimal_lat = dms_to_decimal(*lat_parts, lat_ref)
        decimal_lon = dms_to_decimal(*lon_parts, lon_ref)
        print(f"Extracted GPS Coordinates: Latitude {decimal_lat}, Longitude {decimal_lon}")

        return decimal_lon, decimal_lat

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None


# Recursive function to match ground truth data with the closest image
def match_ground_truth_recursive(ground_truth_data, images_dir, index=0, matches=[]):
    if index >= len(ground_truth_data):
        print("All ground truth entries processed.")
        return matches

    row = ground_truth_data.iloc[index]
    gt_lat, gt_lon = row['Latitude'], row['Longitude']

    if pd.isna(gt_lat) or pd.isna(gt_lon):
        print(f"Skipping invalid ground truth entry at index {index}: ({gt_lat}, {gt_lon})")
        return match_ground_truth_recursive(ground_truth_data, images_dir, index + 1, matches)

    print(f"Processing ground truth entry at index {index}: ({gt_lat}, {gt_lon})")

    closest_img = None
    closest_distance = float('inf')

    # Loop through each image to find the closest match
    for filename in os.listdir(images_dir):
        if filename.lower().endswith('.tif'):
            img_path = os.path.join(images_dir, filename)
            print(f"Checking image: {filename}")

            img_gps = extract_gps_from_exif(img_path)

            if img_gps is None:
                print(f"No valid GPS data in {filename}. Skipping.")
                continue

            img_lon, img_lat = img_gps  # Switched lon, lat ordering for comparison

            # Calculate distance if both coordinates are valid
            try:
                distance = geodesic((gt_lat, gt_lon), (img_lat, img_lon)).meters  # Distance in meters
                print(f"Distance from ground truth to {filename}: {distance:.2f} meters")

                if distance < closest_distance:
                    closest_distance = distance
                    closest_img = filename
                    print(f"New closest image: {closest_img} with distance {closest_distance:.2f} meters")

            except ValueError as ve:
                print(f"Invalid coordinates: Ground Truth ({gt_lat}, {gt_lon}), Image ({img_lat}, {img_lon})")
                continue

    # Save the closest image match for this ground truth point
    if closest_img:
        matches.append({
            "GroundTruth_Lat": gt_lat,
            "GroundTruth_Lon": gt_lon,
            "ImageFile": closest_img,
            "Distance": closest_distance
        })
        print(f"Match found for index {index}: {closest_img} with distance {closest_distance:.2f} meters")

    return match_ground_truth_recursive(ground_truth_data, images_dir, index + 1, matches)

# Load ground truth data
ground_truth_path = 'C:/Users/Jahin Catalan Mahbub/My Drive (mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/StrawberryNDVI_ChlorophyllData_Cristobal/09202024 strawberry plot, ndvi chlorophyll(09202024 strawberry ndvi) (1).csv'
ground_truth_data = pd.read_csv(ground_truth_path, skiprows=4)  # Adjust rows if needed
ground_truth_data.columns = ['Longitude', 'Latitude', 'No', 'Red', 'NIR', 'NDVI', 'Plant_health', 'Chlorophyll']

# Directory with .tif images
images_dir = 'M:/Workz/CPP Canvas/CS6910RA/CPP_Drone_Strawberry_F23/20240920_Strawberry_Multispectral'

# Run recursive matching process
print("Starting recursive ground truth matching...")
matches = match_ground_truth_recursive(ground_truth_data, images_dir)

# Convert matches to DataFrame for analysis
matches_df = pd.DataFrame(matches)
print("Final matches DataFrame:")
print(matches_df)