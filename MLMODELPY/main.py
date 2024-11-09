import subprocess
import os
import re


def dms_to_decimal(degrees, minutes, seconds, direction):
    """Convert coordinates from degrees/minutes/seconds to decimal format"""
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal


def extract_gps_from_exif(filepath):
    """Extract GPS coordinates from image using exiftool"""
    try:
        # Run exiftool and capture output
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


def main():
    # Get current directory
    folder_path = r'M:\Workz\CPP Canvas\CS6910RA\CPP_Drone_Strawberry_F23\20240920_Strawberry_Multispectral'

    # Print header
    print("Longitude,Latitude")

    # Process all TIF files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif'):
            filepath = os.path.join(folder_path, filename)
            coordinates = extract_gps_from_exif(filepath)

            if coordinates:
                lon, lat = coordinates
                print(f"{lon:.7f},{lat:.7f}")


if __name__ == "__main__":
    main()