import geopandas as gpd

# Load your boundary file
boundary = gpd.read_file('data/raw/Prades/boundry1.gpkg')

print("Boundary Information:")
print(f"  CRS: {boundary.crs}")
print(f"  Number of features: {len(boundary)}")
print(f"  Geometry type: {boundary.geometry.iloc[0].geom_type}")
print(f"  Bounds: {boundary.geometry.iloc[0].bounds}")
print(f"  Area: {boundary.geometry.iloc[0].area / 1_000_000:.2f} km²")
print(f"  Is valid: {boundary.geometry.iloc[0].is_valid}")

# Check if it overlaps with your data
print("\nChecking overlap with Dataset1:")
from shapely.geometry import Point

# Test with a point from your data range
test_point = Point(780000, 6288000)
print(f"  Test point (780000, 6288000):")
print(f"    Within boundary: {test_point.within(boundary.geometry.iloc[0])}")

# Get actual bounds
print(f"\n  Boundary X range: {boundary.geometry.iloc[0].bounds[0]:.0f} to {boundary.geometry.iloc[0].bounds[2]:.0f}")
print(f"  Boundary Y range: {boundary.geometry.iloc[0].bounds[1]:.0f} to {boundary.geometry.iloc[0].bounds[3]:.0f}")