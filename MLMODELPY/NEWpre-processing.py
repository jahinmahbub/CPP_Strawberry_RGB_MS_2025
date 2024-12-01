import os
import re
import subprocess
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support

def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees."""
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps_from_exif(filepath):
    """Extract GPS coordinates from EXIF metadata for .jpg images."""
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


def map_plants_to_images_with_health(ground_truth, image_data, distance_threshold=10.0):
    """
    Map plants to images within a distance threshold and include health status.
    """
    print("Calculating distances...")
    results = {}
    for image_file, (img_lat, img_lon) in image_data:
        img_lon = -abs(img_lon)  # Ensure longitude is negative
        visible_plants = []
        for plant in ground_truth:
            gt_lat = plant['Latitude']
            gt_lon = plant['Longitude']
            distance = calculate_distance(gt_lat, gt_lon, img_lat, img_lon)
            if distance <= distance_threshold:
                # Determine health status based on Plant_health
                health = "Healthy" if plant['Plant_health'] >= 3 else "Unhealthy"
                plant_info = f"N={plant['No']}|{health}"
                visible_plants.append(plant_info)

        # Only add entries if plants are visible in the image
        if visible_plants:
            results[image_file] = visible_plants
    return results


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points."""
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    
    
def main():
    # Load ground truth data
    ground_truth_path = 'C:/Users/Jahin Catalan Mahbub/My Drive (mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/StrawberryNDVI_ChlorophyllData_Cristobal/09202024 strawberry plot, ndvi chlorophyll(09202024 strawberry ndvi) (1).csv'
    print("Loading ground truth data...")
    ground_truth_data = pd.read_csv(ground_truth_path, skiprows=4)
    ground_truth_data.columns = ['Longitude', 'Latitude', 'No', 'Red', 'NIR', 'NDVI', 'Plant_health', 'Chlorophyll']
    ground_truth = ground_truth_data.dropna(subset=['Latitude', 'Longitude']).to_dict(orient="records")

    # Directory with .jpg images
    images_dir = 'M:/Workz/CPP Canvas/CS6910RA/CPP_Drone_Strawberry_F23/20240920_Strawberry_Multispectral'
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]

    print("Extracting GPS data from RGB images...")
    image_data = []
    for img_file in image_files:
        gps = extract_gps_from_exif(img_file)
        if gps:
            image_data.append((img_file, gps))

    print(f"Extracted GPS data for {len(image_data)} images.")

    # Map plants to images with health status
    print("Mapping plants to images with health status...")
    distance_threshold = 10.0  # Adjust as needed
    plant_image_mapping = map_plants_to_images_with_health(ground_truth, image_data, distance_threshold)

    # Save results to CSV
    output_path = 'rgb_image_plant_health_mapping.csv'
    with open(output_path, 'w') as f:
        f.write("ImageFile,Plant_Nos,Health\n")
        for image_file, plant_list in plant_image_mapping.items():
            plants = ','.join(plant_list)
            f.write(f"{os.path.basename(image_file)},{plants}\n")

    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    freeze_support()  # Required for Windows
    main()
