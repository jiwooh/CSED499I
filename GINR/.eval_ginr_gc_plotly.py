"""
Evaluates a trained model by predicting on one sample from the specified dataset.

Arguments:
    - checkpoint: path to a Pytorch Lightning checkpoint file
    - idx: integer index of a sample from the dataset to predict and plotting
    - mesh: if not None, a path to a mesh file (obj, ply, etc.) on which to plotting the
            predicted signal. If None, the signal is plotted as a scatter plotting.
    - batch_size: number of points per batch when predicting with the trained model
                  (higher is better)

Note: requires the --dataset_dir flag to be specified as well.
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
from plotly import graph_objects as go

from src.data.graph_dataset import GraphDataset
from src.models.graph_inr import GraphINR
from src.plotting import traces
from src.plotting.figures import PLOT_CONFIGS, draw_mesh
from src.utils.data_generation import load_mesh
from src.utils.get_predictions import get_batched_predictions

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

pl.seed_everything(1234)

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--idx", default=0, type=int)
parser.add_argument("--mesh", default=None, type=str)
parser.add_argument("--key", type=str, default="gc")
parser.add_argument("--batch_size", default=1000000, type=int)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

# assert args.key in [
#     "bunny",
#     "protein_1AA7_A",
# ], '--key must be in ["bunny", "protein_1AA7_A"]'

# Model
model = GraphINR.load_from_checkpoint(args.checkpoint)

# Data
dataset = GraphDataset(**vars(args))
# points = dataset.npzs[0]["points"]
points = np.load(os.path.join(dataset.dataset_dir, "points.npy"))
inputs = dataset.get_inputs(0)

# Predict
_, pred = get_batched_predictions(model, inputs, 0, batch_size=args.batch_size)

if args.mesh is None:
    mesh = load_mesh(args.mesh)
    key = "gc" #args.key
    rot = R.from_euler("xyz", [90, 00, 145], degrees=True).as_matrix() #PLOT_CONFIGS[key]["rot"]
    fig = draw_mesh(
        mesh,
        intensity=pred,
        rot=rot,
        colorscale="Blues", #PLOT_CONFIGS[key]["colorscale"],
        lower_camera=True #PLOT_CONFIGS[key]["lower_camera"],
    )
    fig.show()
    fig.write_html(f"{args.key}.html")
    

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

# Create a grid of latitudes and longitudes
lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)

# Flatten the grid to get the lat, lon pairs
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()

# Convert lat/lon to Cartesian coordinates (x, y, z)
x, y, z = lat_lon_to_cartesian(lat_flat, lon_flat)

# Assuming pred contains the predicted values (e.g., geopotential, temperature, etc.)
# Example prediction (replace this with actual predicted values)
pred = np.random.random(len(x))  # Replace with actual predictions

# Create the plotly scatter3d plot
fig = go.Figure()

# Add scatter plot on the sphere
fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=pred,  # Color based on predicted values
        colorscale='Viridis',  # Color scale for predicted values
        colorbar=dict(title="Prediction Intensity"),
    ),
))

# Update layout
fig.update_layout(
    title="3D Plot of Predictions on a Sphere",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'  # Maintain aspect ratio
    ),
)

# # Show the plot in the browser
# fig.show()

# # Save plot to html file
# fig.write_html("sphere_predictions.html")

# Save the plot as a PNG file (static image)
fig.write_image("sphere_predictions.png", width=1920, height=1080)

print("PNG image saved as sphere_predictions.png")