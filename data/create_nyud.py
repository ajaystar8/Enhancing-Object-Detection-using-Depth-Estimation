"""
Script for Processing and Saving NYU Depth V2 Dataset

This script processes the NYU Depth V2 dataset stored in a .mat file and saves the images, depth maps, and raw depth maps
as PNG files. The script performs the following operations:

1. Reads the .mat file and extracts datasets for RGB images, depth maps, and raw depth maps.
2. Creates directories to store processed images:
   - ../NYUD/images: For RGB images
   - ../NYUD/depth: For depth maps
   - ../NYUD/raw_depth: For raw depth maps
3. Processes each dataset:
   - Converts RGB images from the original format to PNG format with pixel values normalized to 16-bit integers.
   - Converts depth maps and raw depth maps to PNG format with values normalized to 16-bit integers.
   - Handles image rotation and flipping to ensure correct orientation.
4. Saves the processed data in the respective directories.

Usage:
Ensure the .mat file is located at the specified `filename` path. Run the script to process the dataset and save the outputs.
"""

import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

filename = "../resources/nyu_depth_v2_labeled.mat"

data = None
# Open the .mat file
with h5py.File(filename, "r") as mat_file:
	data = {key: np.array(mat_file[key]) for key in mat_file.keys()}

	# Inspect the keys
	print("Keys in the file:", list(mat_file.keys()))

	# Access a dataset
	images = mat_file['images']  # Replace 'images' with the actual key in your file
	depths = mat_file['depths']  # Replace with actual key
	raw_depths = mat_file['rawDepths']

	# Convert datasets to NumPy arrays
	images = images[()]  # If images is a dataset, this retrieves it as a NumPy array
	depths = depths[()]
	raw_depths = raw_depths[()]

os.makedirs(os.path.join("..", "NYUD"), exist_ok=True)
os.makedirs(os.path.join("..", "NYUD", "images"), exist_ok=True)
os.makedirs(os.path.join("..", "NYUD", "depth"), exist_ok=True)
os.makedirs(os.path.join("..", "NYUD", "raw_depth"), exist_ok=True)

for idx in tqdm(range(images.shape[0])):
	image_array = np.fliplr(np.rot90(np.transpose(images[idx], (1, 2, 0)), k=-1))
	normalized_image_array = np.uint8(
		(image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 65535)
	image = Image.fromarray(normalized_image_array)
	image.save(os.path.join("..", "NYUD", "images", f"{idx}.png"))

print("All raw depth maps saved as png images!")

for idx in tqdm(range(depths.shape[0])):
	depth_map = np.fliplr(np.rot90(depths[idx], k=-1))
	normalized_depth_map = np.uint16((depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 65535)
	depth_image = Image.fromarray(normalized_depth_map)
	depth_image.save(os.path.join("..", "NYUD", "depth", f"{idx}.png"))

print("All raw depth maps saved as png images!")

for idx in tqdm(range(raw_depths.shape[0])):
	raw_depth_map = np.fliplr(np.rot90(raw_depths[idx], k=-1))
	# depth_map = depths[idx]
	normalized_raw_depth_map = np.uint16(
		(raw_depth_map - np.min(raw_depth_map)) / (np.max(raw_depth_map) - np.min(raw_depth_map)) * 65535)
	raw_depth_image = Image.fromarray(normalized_raw_depth_map)
	raw_depth_image.save(os.path.join("..", "NYUD", "raw_depth", f"{idx}.png"))

print("All raw depth maps saved as png images!")
