import folium
import pandas as pd


def load_and_clean_gt_data(gt_csv_path):
	"""
	Load and clean the ground truth dataset.
	Args:
		gt_csv_path (str): Path to the GT CSV file.

	Returns:
		pd.DataFrame: Filtered dataset with valid Longitude and Latitude values.
	"""
	# Load the CSV file, skipping metadata rows
	data = pd.read_csv(gt_csv_path, skiprows=4)
	data.columns = [col.strip() for col in data.columns]  # Clean column names
	# Filter out rows where Longitude or Latitude is NaN
	return data.dropna(subset=["Longitude", "Latitude"])


# Main function
if __name__ == "__main__":
	# Path to your GT dataset CSV file
	gt_csv_path = 'C:/Users/Jahin Catalan Mahbub/My Drive (mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/StrawberryNDVI_ChlorophyllData_Cristobal/09202024 strawberry plot, ndvi chlorophyll(09202024 strawberry ndvi) (1).csv'
	
	# Load and clean the GT data
	filtered_data = load_and_clean_gt_data(gt_csv_path)
	
	# Calculate the map center
	center_lat = filtered_data["Latitude"].mean()
	center_lon = filtered_data["Longitude"].mean()
	
	# Create a folium map centered at the calculated location
	m = folium.Map(location=[center_lat, center_lon], zoom_start=20, tiles='OpenStreetMap')
	
	# Add GT points as markers
	for _, row in filtered_data.iterrows():
		folium.CircleMarker(
			location=(row["Latitude"], row["Longitude"]),
			radius=3,
			color="red",
			fill=True,
			fill_color="red"
		).add_to(m)
	
	# Save the map to an HTML file
	m.save("strawberry_field_map.html")
	print("Interactive map saved as 'strawberry_field_map.html'. Open it in a browser to view.")