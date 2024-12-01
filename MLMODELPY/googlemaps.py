import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds


def fetch_satellite_image(center, zoom, size, api_key):
	"""
    Fetch a satellite image using Google Maps Static API.
    Args:
        center (tuple): (latitude, longitude) of the center point.
        zoom (int): Zoom level for the map.
        size (tuple): Size of the image (width, height) in pixels.
        api_key (str): Google Maps API key.
    Returns:
        PIL.Image: Satellite image.
    """
	url = (
		f"https://maps.gomaps.pro/maps/api/staticmap?center={center[0]},{center[1]}"
		f"&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&key={api_key}"
	)
	response = requests.get(url)
	response.raise_for_status()
	image = Image.open(BytesIO(response.content))
	
	# Convert the image to RGB mode
	if image.mode != "RGB":
		image = image.convert("RGB")
	
	return image


def create_geotiff(image, bounds, output_path):
	"""
	Save the image as a GeoTIFF with geocoordinates.
	Args:
		image (PIL.Image): Image to georeference.
		bounds (tuple): (min_lon, max_lon, min_lat, max_lat) bounding box.
		output_path (str): Path to save the GeoTIFF.
	"""
	if image.mode != "RGB":
		image = image.convert("RGB")
	
	image_array = np.array(image)
	height, width, _ = image_array.shape
	transform = from_bounds(*bounds, width, height)
	
	with rasterio.open(
			output_path,
			"w",
			driver="GTiff",
			height=height,
			width=width,
			count=3,  # RGB image
			dtype="uint8",
			crs="EPSG:4326",  # WGS 84
			transform=transform,
	) as dst:
		dst.write(image_array[:, :, 0], 1)
		dst.write(image_array[:, :, 1], 2)
		dst.write(image_array[:, :, 2], 3)


def draw_rectangle_and_points(image, bounds, rectangle_coords, gt_csv):
	"""
	Draw a rectangle and mark GT points on the image.
	Args:
		image (PIL.Image): Image to modify.
		bounds (tuple): (min_lon, max_lon, min_lat, max_lat) bounding box.
		rectangle_coords (list): List of (lat, lon) tuples defining the rectangle.
		gt_csv (str): Path to the GT CSV file.
	Returns:
		PIL.Image: Modified image with rectangle and points.
	"""
	draw = ImageDraw.Draw(image)
	width, height = image.size
	min_lon, max_lon, min_lat, max_lat = bounds
	
	# Draw rectangle
	rectangle_pixels = [
		((lon - min_lon) / (max_lon - min_lon) * width, (max_lat - lat) / (max_lat - min_lat) * height)
		for lat, lon in rectangle_coords
	]
	draw.polygon(rectangle_pixels, outline="yellow", width=3)
	
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
	
	return image


if __name__ == "__main__":
	# Google Maps API key
	API_KEY = "AlzaSyoxgLN0e601SvjR4b8IFfFUPEYkipZFgqm"  # Replace with your API key
	# Define the four corners of the rectangle
	rectangle_coords = [
		(34.043713, -117.811502),  # Top-left corner
		(34.043539, -117.811236),  # Top-right corner
		(34.042998, -117.811768),  # Bottom-right corner
		(34.043168, -117.812034)  # Bottom-left corner
	]
	
	# Calculate bounding box
	min_lon = min(lon for lat, lon in rectangle_coords)
	max_lon = max(lon for lat, lon in rectangle_coords)
	min_lat = min(lat for lat, lon in rectangle_coords)
	max_lat = max(lat for lat, lon in rectangle_coords)
	bounds = (min_lon, max_lon, min_lat, max_lat)
	
	# Center of the rectangle
	center = ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
	
	# Fetch the satellite image
	try:
		zoom = 21  # Adjust zoom level for desired resolution
		dim = 1230
		size = (dim, dim)  # Size of the image in pixels
		satellite_image = fetch_satellite_image(center, zoom, size, API_KEY)
		print("Satellite image successfully fetched.")
		
		# Draw rectangle and GT points
		gt_csv_path = 'C:/Users/Jahin Catalan Mahbub/My Drive (mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/StrawberryNDVI_ChlorophyllData_Cristobal/09202024 strawberry plot, ndvi chlorophyll(09202024 strawberry ndvi) (1).csv'  # Replace with the path to your GT CSV file
		modified_image = draw_rectangle_and_points(satellite_image, bounds, rectangle_coords, gt_csv_path)
		
		# Save the modified image as GeoTIFF
		output_path = "strawberry_field_with_points.tif"
		create_geotiff(modified_image, bounds, output_path)
		print(f"GeoTIFF saved at: {output_path}")
	
	except Exception as e:
		print(f"Error: {e}")