import xarray as xr
import numpy as np
from plyfile import PlyData, PlyElement
import plotly.graph_objects as go

### SELECTION ###
vardata = [
    '10m_u_component_of_wind', # 0
    '10m_v_component_of_wind', # 1
    '2m_temperature',          # 2
    'mean_sea_level_pressure', # 3
    'total_precipitation_6hr', # 4
    'geopotential',            # 5
    'specific_humidity',       # 6
    'temperature',             # 7
    'u_component_of_wind',     # 8
    'v_component_of_wind',     # 9
    'vertical_velocity',       # 10
]
# 3, 4, 6 or 7
_VAR_INDEX = 3

# 5 or 6
_MESH_SIZE = 6
### SELECTION ###

dataset = xr.open_dataset(f'dataset/predictions_m{_MESH_SIZE}_g4_l32.nc')

# pick only 1 time entry
dataset = dataset.isel(time=0)

# pick only level 1000
dataset = dataset.isel(level=-1)

# drop time and level coordataset
dataset = dataset.drop_vars(["time", "level"])

# Extract lat, lon, and variable data
lat = dataset['lat'].values  # Shape (721,)
lon = dataset['lon'].values  # Shape (1440,)
var = dataset[vardata[_VAR_INDEX]][0, :, :].values  # Shape (721, 1440), drop batch dimension

# Generate the grid for lat and lon (meshgrid)
lat_grid, lon_grid = np.meshgrid(lat, lon)

# Flatten the lat, lon, and var grids to create the vertices
vertices = np.vstack((lat_grid.flatten(), lon_grid.flatten(), var.flatten())).T

# Create the header for the .ply file
n_vertices = vertices.shape[0]
header = f"""ply
format ascii 1.0
element vertex {n_vertices}
property float x
property float y
property float z
end_header
"""

# Write the vertices to a .ply file
with open(f'./dataset/gcm{_MESH_SIZE-1}to{_MESH_SIZE}/mesh_{vardata[_VAR_INDEX]}.ply', 'w') as f:
    f.write(header)
    for v in vertices:
        f.write(f"{v[0]} {v[1]} {v[2]}\n")

print(f"PLY file of mesh size {_MESH_SIZE} with data {vardata[_VAR_INDEX]} created successfully.")