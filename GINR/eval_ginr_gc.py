"""
Evaluates a trained model by predicting on one sample from the specified dataset.

Arguments:
    - checkpoint: path to a Pytorch Lightning checkpoint file
    - idx: integer index of a sample from the dataset to predict and plotting
    - variable: predicted variable to plot
    - batch_size: number of points per batch when predicting with the trained model
                  (higher is better)

Note: requires the --dataset_dir flag to be specified as well.
"""

varlist = [
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

_SAVE_PRED_DATA = not False

from argparse import ArgumentParser

import os
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from src.data.graph_dataset import GraphDataset
from src.models.graph_inr import GraphINR
from src.utils.get_predictions import get_batched_predictions

pl.seed_everything(1234)

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--idx", default=0, type=int)
parser.add_argument("--variable", default=7, type=int)
parser.add_argument("--key", type=str, default="gc")
parser.add_argument("--batch_size", default=1000000, type=int)
parser.add_argument("--filename", default="filename", type=str)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

_VAR_INDEX = args.variable
_MESH_SIZE = args.dataset_dir[-1]
_FILENAME = args.filename

print(f"INFO: data = {varlist[_VAR_INDEX]}, varlist.index = {_VAR_INDEX}, mesh = {_MESH_SIZE}, filename = {_FILENAME}")

# Model
model = GraphINR.load_from_checkpoint(args.checkpoint)

# Data
dataset = GraphDataset(**vars(args))
# points = dataset.npzs[0]["points"]
points = np.load(os.path.join(dataset.dataset_dir, "points.npy"))
inputs = dataset.get_inputs(0)

# Prediction and Data Preparation
print("INFO: predicting")
_, pred = get_batched_predictions(model, inputs, 0, batch_size=args.batch_size)

def lat_lon_to_cartesian(lat, lon, radius=1):
    """
    Converts latitude and longitude to Cartesian coordinates on a sphere.

    Arguments:
    - lat: Latitude in degrees.
    - lon: Longitude in degrees.
    - radius: Radius of the sphere (default is 1 for unit sphere).

    Returns:
    - A tuple (x, y, z) of Cartesian coordinates.
    """
    # Convert lat and lon from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Spherical to Cartesian conversion
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z

latitudes = np.unique(points[:,0])
longitudes = np.unique(points[:,1])

# Create a meshgrid of latitudes and longitudes
lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)

# Flatten the grid to get latitudes and longitudes as 1D arrays
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()

# Convert lat/lon to Cartesian coordinates (x, y, z)
x, y, z = lat_lon_to_cartesian(lat_flat, lon_flat)

# Get predictions
_, pred = get_batched_predictions(model, inputs, 0, batch_size=args.batch_size)
if _SAVE_PRED_DATA: 
  np.save(f'./results/{_FILENAME}.npy', pred)
  print(f"INFO: prediction saved as ./results/{_FILENAME}.npy")

# Select variable data
pred = pred[:,_VAR_INDEX:_VAR_INDEX+1]

assert len(pred) == len(x), f"Pred array length does not match number of data points. Found {len(pred)} but expected {len(x)}."

# Visualization
print("INFO: visualizing")
# Create the figure and 3D axes
fig = plt.figure(figsize=(12, 4), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

# Scatter plot on the 3D plot
sc = ax.scatter(x, y, z, c=pred, cmap='plasma', s=5)

# Add a colorbar to represent the predicted values
cbar = fig.colorbar(sc, ax=ax, label='Prediction Intensity', fraction=0.015, pad=0.04)
# cbar.ax.tick_params(labelsize=8)

# Set title (adding the predicted variable)
ax.set_title(f"Predicted {varlist[_VAR_INDEX]}")
# plt.title(_FILENAME)
ax.text2D(0.5, 0.92, _FILENAME, transform=ax.transAxes, ha="center", fontsize=8, color="gray")


# Add grid lines to visualize the Earth-like distribution
ax.grid(True)

# Set the plot's aspect ratio to be equal (e.g., spherical Earth visualization)
ax.set_box_aspect([1, 1, 1])  # Equal scaling for X, Y, Z axes

# Remove ticks and background
ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks
ax.set_zticks([])  # Remove z-axis ticks

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Save the figure as a PNG image
plt.savefig(f'./results/{_FILENAME}.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"PNG image saved as ./results/{_FILENAME}.png")
