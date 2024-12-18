import glob
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import scipy.io

"""
1. ultrasound Image Read in as RGB / Gray
2. Transducer Location: 2 orientation: a) object x location x xy-coordinate; b) object x location x xy-coordinate
3. Result map

"""

def get_paths(root_path):
    # given path to dataset (contain sub directory images, masks, simulation_outputs)
    images_path = glob.glob(os.path.join(f"{root_path}".format(data_type = 'images'), '*'))
    images_path.sort()
    masks_path = glob.glob(os.path.join(f"{root_path}".format(data_type = 'masks'), '*'))
    masks_path.sort()
    simulation_path = glob.glob(os.path.join((root_path).format(data_type = 'simulation_outputs'), '*_max_pressure.mat'), recursive=True)
    simulation_path.sort()
    print('image path', (f"{root_path}/*").format(data_type = 'images'))
    print('masks path', (f"{root_path}/*").format(data_type = 'masks'))
    print('simulation path', (f"{root_path}/*").format(data_type = 'simulation_outputs'))
    #return images_path, simulation_path, 
    return masks_path, simulation_path #use masks as input

class TransducerDataset(Dataset):
    def __init__(
        self, 
        image_path, 
        simulation_path, 
        image_transforms = True,
        loading_method = 'individual',
        device = 'cpu'
    ):

        self.image_paths = image_path
        self.simulation_paths = simulation_path
        self.image_transforms = image_transforms
        self.loading_method = loading_method
        self.device  = device


        # #Image width and Height
        self.width = 512
        self.height = 162

        self.sim_width = 512
        self.sim_height = 162
   
        
        transducer_locs=np.array([[0]*8,[40.,	101.,	163.,	225.,	286.,	348.,	410.,	472.]]) # FOR LOW RES


        transducer_locs = transducer_locs.transpose((1,0))
        self.transducer_locs = torch.tensor(transducer_locs, dtype=torch.float32).to(device)

        x = np.arange(0, self.sim_height)
        y = np.arange(0, self.sim_width)
        xx, yy = np.meshgrid(x, y)
        locations = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        self.sensor_locations = torch.tensor(locations, dtype=torch.float).to(self.device)

        #Preload
        self.preloaded_images = [self.image_load(i).to(device) for i in range(len(self.image_paths))]
        self.preloaded_simulations = [self.simulation_indv_load(i).to(device) for i in range(len(self.simulation_paths))]
        
    def get_sensor_location(self):
        # Generate array of coordinates / sensor
        return

    def __len__(self):
        if self.loading_method =='individual':
            return len(self.simulation_paths) # image, tran loc, simu -> 1 sample
        elif self.loading_method == 'group':
            return len(self.image_paths) # image, 8 tran loc, 8 simu -> 1 sample
        elif self.loading_method[:3] == 'loc':
            return len(self.image_paths) # image, 1 tran loc, 1 simu -> 1 sample
        else:
            print("can't recognize loading method")

    def set_dim(self, new_img_height:int, new_img_width:int):
        self.height = new_img_height
        self.width = new_img_width
        if self.image_transforms:
            print(f"will re-sample the image to {new_img_height} x {new_img_width}")
        else:
            print(f'Resized. But current dataset is NOT configed to resample image')
        
    def simulation_indv_load(self, index):
        # Process Simulations Individually
        simulations = scipy.io.loadmat(self.simulation_paths[index])['p_max']
        simulations = torch.tensor(simulations, dtype=torch.float) 
        self.sim_height = simulations.shape[-2]
        self.sim_width = simulations.shape[-1]
        return simulations

    def simulation_group_load(self, index): # AVISHA 
        simulations = []
        for i in range(index*8, (index+1)*8):
            simulation = scipy.io.loadmat(self.simulation_paths[i])['p_max']
            simulations.append(simulation)
        simulations = np.array(simulations)
        simulations = torch.tensor(simulations, dtype=torch.float) 
        self.sim_height = simulations.shape[-2] #update simulation location coords based on simulation lable size
        self.sim_width = simulations.shape[-1]
        return simulations
    
    def image_load(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB') 

        if self.image_transforms:
            image_transforms = transforms.Compose([
                transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),  # resample image
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = image_transforms(image)
        else:
            image = np.array(image)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
        
        return image

    def eval_locs(self, type = 'sq'):
        #Linear
        self.get_sensor_location() #cal func to generate sensor location based on simulation
        if type == 'lin':
            locations = self.sensor_locations # array (x,y)
        #Square
        elif type == 'sq':
            locations = self.sensor_locations.reshape(self.sim_width,self.sim_height,2).permute(1,0,2) # array of array of (x,y )
        #Cubic
        else:
            ...
        return locations 

    def __getitem__(self, index):
        if self.loading_method =='individual':
            #Images
            image = self.preloaded_images[index//8]

            #Transducer location
            transducer_locs = self.transducer_locs[index%8]

            #evaludation location (sensor for deepOnet)
            locs = self.eval_locs(type = 'sq')

            #Simulation
            simulations = self.preloaded_simulations[index]

        elif self.loading_method[:3] == 'loc':
            loc_index = int(self.loading_method[-1])
            #Images
            image = self.preloaded_images[index]
            #Transducer location
            transducer_locs = self.transducer_locs[loc_index]
            
            #evaludation location (sensor for deepOnet)
            locs = self.eval_locs(type = 'sq')

            #Simulation
            simulations = self.preloaded_simulations[index*8 + loc_index]
        else:
            ...

        return image, transducer_locs, locs, simulations