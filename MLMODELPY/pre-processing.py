import geopandas as gpd
import pandas as pd

# Load each GeoJSON file directly with gpd.read_file()
gndvi_data = gpd.read_file("C:\\Users\\mzahi\\My Drive\\CPP Canvas\\CS6910RA\\MLMODELPY\\10112024\\10112024_MS\\10112024_MS_GNDVIZonation.geojson")
mcari_data = gpd.read_file("C:\\Users\\mzahi\\My Drive\\CPP Canvas\\CS6910RA\\MLMODELPY\\10112024\\10112024_MS\\10112024_MS_MCARIZonation.geojson")
ndvi_data = gpd.read_file("C:\\Users\\mzahi\\My Drive\\CPP Canvas\\CS6910RA\\MLMODELPY\\10112024\\10112024_MS\\10112024_MS_NDVIZonation.geojson")
tgi_data = gpd.read_file("C:\\Users\\mzahi\\My Drive\\CPP Canvas\\CS6910RA\\MLMODELPY\\10112024\\10112024_MS\\10112024_MS_TGIZonation.geojson")
vari_data = gpd.read_file("C:\\Users\\mzahi\\My Drive\\CPP Canvas\\CS6910RA\\MLMODELPY\\10112024\\10112024_MS\\10112024_MS_VARIZonation.geojson")

# Inspect data to understand structure
print(gndvi_data.head())
print(mcari_data.head())
print(ndvi_data.head())
print(tgi_data.head())
print(vari_data.head())

# Print column names for each GeoDataFrame
print("GNDVI columns:", gndvi_data.columns)
print("MCARI columns:", mcari_data.columns)
print("NDVI columns:", ndvi_data.columns)
print("TGI columns:", tgi_data.columns)
print("VARI columns:", vari_data.columns)


# Rename 'avgValue' in each dataset to its respective index name
gndvi_data = gndvi_data.rename(columns={'avgValue': 'GNDVI'})
mcari_data = mcari_data.rename(columns={'avgValue': 'MCARI'})
ndvi_data = ndvi_data.rename(columns={'avgValue': 'NDVI'})
tgi_data = tgi_data.rename(columns={'avgValue': 'TGI'})
vari_data = vari_data.rename(columns={'avgValue': 'VARI'})

# Ensure all datasets are in the same CRS
crs = gndvi_data.crs  # Assuming CRS is the same, take from any file, e.g., GNDVI
mcari_data = mcari_data.to_crs(crs)
ndvi_data = ndvi_data.to_crs(crs)
tgi_data = tgi_data.to_crs(crs)
vari_data = vari_data.to_crs(crs)

# Reset index to avoid 'index_right' conflicts
gndvi_data = gndvi_data.reset_index(drop=True)
mcari_data = mcari_data.reset_index(drop=True)
ndvi_data = ndvi_data.reset_index(drop=True)
tgi_data = tgi_data.reset_index(drop=True)
vari_data = vari_data.reset_index(drop=True)

# Perform spatial joins to combine data, removing 'index_right' if it appears
merged_data = gpd.sjoin(gndvi_data[['geometry', 'GNDVI']], mcari_data[['geometry', 'MCARI']], how="inner", predicate="intersects")
if 'index_right' in merged_data.columns:
    merged_data = merged_data.drop(columns=['index_right'])

merged_data = gpd.sjoin(merged_data, ndvi_data[['geometry', 'NDVI']], how="inner", predicate="intersects")
if 'index_right' in merged_data.columns:
    merged_data = merged_data.drop(columns=['index_right'])

merged_data = gpd.sjoin(merged_data, tgi_data[['geometry', 'TGI']], how="inner", predicate="intersects")
if 'index_right' in merged_data.columns:
    merged_data = merged_data.drop(columns=['index_right'])

merged_data = gpd.sjoin(merged_data, vari_data[['geometry', 'VARI']], how="inner", predicate="intersects")
if 'index_right' in merged_data.columns:
    merged_data = merged_data.drop(columns=['index_right'])

# Drop duplicate columns created by spatial join
merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

# Set display option to avoid truncation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full column width

# Print the first 5 rows of the merged data
print(merged_data.head(5))

# Reset the display options if needed
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')

# Group by unique geometry and aggregate index columns by mean
merged_data = merged_data.groupby('geometry', as_index=False).agg({
    'GNDVI': 'mean',
    'MCARI': 'mean',
    'NDVI': 'mean',
    'TGI': 'mean',
    'VARI': 'mean'
})

# Export the merged data to a CSV file
merged_data.to_csv("merged_plant_health_data.csv", index=False)

print("Data has been saved to 'merged_plant_health_data.csv'")
# Load the data from the CSV
data = pd.read_csv("merged_plant_health_data.csv")

# Drop the geometry column
data_no_geometry = data.drop(columns=["geometry"])

# Save this cleaned dataset to a new CSV file
data_no_geometry.to_csv("plant_health_data_nogeometry.csv", index=False)

print("Data has been saved to 'plant_health_data_nogeometry.csv' without geometry column.")