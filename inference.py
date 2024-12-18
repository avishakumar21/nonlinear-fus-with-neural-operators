import torch
import time
from opnn import opnn
import numpy as np
from dataset_prep import get_paths, TransducerDataset
from torch.utils.data import DataLoader
import os
from utils import SegmentationVisualizer  

# model config
EXPECTED_IMG_SIZE = (162, 512)
branch2_dim = [2, 32, 32, 64]  
trunk_dim = [2, 100, 100, 64]  
geometry_dim = EXPECTED_IMG_SIZE

DATA_PATH_IMAGES = r'images\test'
DATA_PATH_SIMULATIONS = r'simulation_outputs\test'
CHECKPOINT_PATH = r'results\model_checkpoint.pth'
RESULTS_FOLDER = r'results\inference'

os.makedirs(RESULTS_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = opnn(branch2_dim, trunk_dim, geometry_dim).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# prepare dataset
image_paths, _ = get_paths(DATA_PATH_IMAGES)  
_, simulation_paths = get_paths(DATA_PATH_SIMULATIONS)  
print(f"Found {len(image_paths)} image files.")
print(f"Found {len(simulation_paths)} simulation files.")

if len(image_paths) == 0 or len(simulation_paths) == 0:
    raise ValueError("No image or simulation files found. Please check the dataset structure or file extensions.")

# test dataset loader
test_dataset = TransducerDataset(image_paths, simulation_paths, loading_method='individual', device=device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
visualizer = SegmentationVisualizer()

for i, (image, transducer_locs, locs, simulations) in enumerate(test_loader):
    print(f"Processing image {i+1}")
    
    image = image.to(device)
    transducer_locs = transducer_locs.to(device)
    locs = locs.to(device)
    simulations = simulations.to(device)
    
    # for timing inference 
    #start_time = time.time()
    with torch.no_grad():
        prediction = model(image, transducer_locs, locs)
    #end_time = time.time()

    # print(f"Time taken for inference on image {i+1}: {end_time - start_time:.6f} seconds")

    #print(f"Prediction shape for image {i+1}: {prediction.shape}")
    images01 = visualizer.minmax_normalize(image)
    simulations01 = visualizer.minmax_normalize(simulations)
    prediction01 = visualizer.minmax_normalize(prediction)

    comment = f'inference_image_{i+1}'
    visualizer.visualize_batch(images01, simulations01, prediction01, batch=1, comment=comment, result_folder=RESULTS_FOLDER)
    print(f"Saved prediction {i+1}")