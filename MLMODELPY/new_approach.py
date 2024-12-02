import rasterio
import pandas as pd
from pyproj import Transformer

# Load GeoTIFF
geo_tiff_path = "Orthomosaic - Boundary.data.tif"
with rasterio.open(geo_tiff_path) as src:
    crs = src.crs  # Coordinate Reference System
    transform = src.transform  # Affine transform
    bounds = src.bounds  # Real-world bounds
    resolution = src.res  # Pixel size (meters/pixel)

# Load GT dataset
gt_csv_path = 'C:/Users/Jahin Catalan Mahbub/My Drive (mzahin.zm@gmail.com)/CPP Canvas/CS6910RA/StrawberryNDVI_ChlorophyllData_Cristobal/09202024 strawberry plot, ndvi chlorophyll(09202024 strawberry ndvi) (1).csv'
gt_data = pd.read_csv(gt_csv_path, skiprows=4).dropna(subset=["Longitude", "Latitude"])
print("GT Data Loaded:", gt_data.head())


# Transformer for CRS conversion
transformer = Transformer.from_crs("EPSG:4326", crs.to_string(), always_xy=True)

# Add transformed coordinates to GT dataset
gt_data["UTM_X"], gt_data["UTM_Y"] = transformer.transform(
    gt_data["Longitude"].values, gt_data["Latitude"].values
)
# Map GT points to GeoTIFF pixels
def map_to_pixel(x, y, transform):
    col = int((x - transform.c) / transform.a)  # X -> Column
    row = int((y - transform.f) / transform.e)  # Y -> Row
    return col, row

gt_data["Pixel_Col"], gt_data["Pixel_Row"] = zip(*[
    map_to_pixel(x, y, transform) for x, y in zip(gt_data["UTM_X"], gt_data["UTM_Y"])
])
print("Mapped Pixels:", gt_data[["Pixel_Col", "Pixel_Row"]].head())
# Extract spectral data from GeoTIFF
with rasterio.open(geo_tiff_path) as src:
    bands = [src.read(i + 1) for i in range(src.count)]  # Read all bands

# Extract values at pixel locations for all bands
gt_data["Band_1"] = [bands[0][row, col] for row, col in zip(gt_data["Pixel_Row"], gt_data["Pixel_Col"])]
gt_data["Band_2"] = [bands[1][row, col] for row, col in zip(gt_data["Pixel_Row"], gt_data["Pixel_Col"])]
gt_data["Band_3"] = [bands[2][row, col] for row, col in zip(gt_data["Pixel_Row"], gt_data["Pixel_Col"])]
gt_data["Band_4"] = [bands[3][row, col] for row, col in zip(gt_data["Pixel_Row"], gt_data["Pixel_Col"])]
gt_data["Band_5"] = [bands[4][row, col] for row, col in zip(gt_data["Pixel_Row"], gt_data["Pixel_Col"])]
# Ignore Band_6 (Alpha) for now

# Compute NDVI
gt_data["NDVI"] = (gt_data["Band_5"] - gt_data["Band_3"]) / (gt_data["Band_5"] + gt_data["Band_3"])

print("GT Data with NDVI:", gt_data[["NDVI"]].head())

print("Number of Bands:", src.count)


output_csv_path = "labeled_gt_dataset.csv"
gt_data.to_csv(output_csv_path, index=False)
print(f"Labeled GT dataset saved at: {output_csv_path}")