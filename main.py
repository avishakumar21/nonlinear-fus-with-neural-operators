import torch
from torch.utils.data import DataLoader,Subset
from torch.optim import Adam
import os
import json
from opnn import opnn
from dataset_prep import get_paths, TransducerDataset
from utils import log_loss, plot_logs,plot_prediction, store_model
from utils import ensure_directory_exists,get_time_YYYYMMDDHH
import argparse
from torch.optim.lr_scheduler import StepLR

## -----  LOAD PARAMETER FROM CONFIG --------- ##

with open('config.json', 'r') as file:
    config = json.load(file)
DATA_PATH = config['DATA_PATH']
RESULT_FOLDER = config['RESULT_FOLDER']
if config['win']:
    DATA_PATH = rf'{DATA_PATH}'
    RESULT_FOLDER = rf'{RESULT_FOLDER}'
epochs = config['epochs']
VIZ_epoch_period = config['VIZ_epoch_period'] 
BATCHSIZE = config['BATCHSIZE']
LR = config['LR']
STEP_SIZE = config['STEP_SIZE']
EXPECTED_IMG_SIZE = config['EXPECTED_IMG_SIZE']
EXPECTED_SIM_SIZE = config['EXPECTED_SIM_SIZE']

class Trainer:
    def __init__(self, model, optimizer, device, num_epochs=1000):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        self.test_loss = None

    def set_result_path (self, result_path):
        self.result_folder = result_path
        self.val_log_path = os.path.join(self.result_folder, 'loss_log_train.txt')
        self.train_log_path = os.path.join(self.result_folder, 'loss_log_val.txt')
        print(f"Result of this Training will be stored at {result_path}")

    def train_one_epoch(self, dataloader):
        self.model.train() 
        total_loss = 0.0 #[]
        num_batches = 0
        for branch1_input, branch2_input, trunk_input, labels in dataloader:
            # Move inputs and labels to the GPU
            branch1_input = branch1_input.to(self.device)
            branch2_input = branch2_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            labels = labels.to(self.device)
            
            # Calculate loss
            loss = self.model.loss(branch1_input, branch2_input, trunk_input, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            #count sample
            num_batches += 1

            #break
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def val_one_epoch(self, dataloader_validation):
        self.model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        num_batches = 0 
        with torch.no_grad():
            for branch1_input, branch2_input, trunk_input, labels in dataloader_validation:
                branch1_input = branch1_input.to(self.device)
                branch2_input = branch2_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                labels = labels.to(self.device)

                val_loss = self.model.loss(branch1_input, branch2_input, trunk_input, labels)
                total_val_loss += val_loss.item()
                num_batches += 1
  
        avg_val_loss = total_val_loss / num_batches
        self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def test(self, dataloader_test, epochs = 0):
        self.model.eval()  # Set the model to evaluation mode
        total_test_loss = 0.0
        num_batches = 0 

        with torch.no_grad():
            for batch, (branch1_input, branch2_input, trunk_input, labels) in enumerate(dataloader_test):
                branch1_input = branch1_input.to(self.device)
                branch2_input = branch2_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                labels = labels.to(self.device)

                test_loss = self.model.loss(branch1_input, branch2_input, trunk_input, labels)
                total_test_loss += test_loss.item()
                num_batches += 1

                #plot sample prediction:
                prediction = self.model(branch1_input, branch2_input, trunk_input)
                plot_prediction(branch1_input.cpu(), labels.cpu(), prediction.cpu(), batch, result_folder=self.result_folder)

        avg_test_loss = total_test_loss / num_batches
        self.test_loss = avg_test_loss
        return avg_test_loss
    
    def visualize_prediction(self, dataloader, comment = '', subset = True):
        if subset:
            subset_indices = list(range(BATCHSIZE))
            subset_dataset = Subset(dataloader.dataset, subset_indices)
            dataloader = DataLoader(subset_dataset, batch_size=BATCHSIZE, shuffle=False)

        self.model.eval()  
        with torch.no_grad():
            for batch, (branch1_input, branch2_input, trunk_input, labels) in enumerate(dataloader):
                prediction = self.model(branch1_input.to(self.device), branch2_input.to(self.device), trunk_input.to(self.device))
                plot_prediction(branch1_input.cpu(), labels.cpu(), prediction.cpu(), batch, comment = comment, result_folder=self.result_folder)
        
        return True


    def train(self, dataloader, dataloader_validation, dataloader_test, scheduler):
        ensure_directory_exists(self.result_folder)

        # Train + Validate Model
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(dataloader)
            scheduler.step()
            val_loss = self.val_one_epoch(dataloader_validation)            
            log_loss(train_loss, temp_file=self.train_log_path)
            log_loss(val_loss, self.val_log_path)
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

            # every X epoch, some sample viz 
            if epoch % VIZ_epoch_period == 0:
                self.visualize_prediction(dataloader_validation, comment = f'val_ep{epoch}',subset=True)

        # Test Model
        if dataloader_test is not None:
            test_loss = self.test(dataloader_test)
            print(f"Test Loss: {test_loss:.4f}")

        # Store model
        store_model(self.model,self.optimizer, epoch, self.result_folder)
        return self.model
    

def load_data_by_split(data_path, bz, shuffle = True):
    print('-'*15, 'DATA READIN BY SPLIT', '-'*15)
    split_path_dict = {}
    for split_name in ['train','val','test']:
        split_data_path=os.path.join(data_path, '{data_type}',split_name)
        images_path,simulation_path = get_paths(split_data_path)
        dataset_ = TransducerDataset(images_path, simulation_path, loading_method='individual')
        dataloader_ = DataLoader(dataset_, batch_size=bz, shuffle=shuffle, num_workers=2)
        split_path_dict[split_name] = dataloader_

    return list(split_path_dict.values())

def main(bz, num_epochs=100, result_folder = RESULT_FOLDER, folder_description = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('-'*15, 'CHECK RESULT DIRECTORY', '-'*15)
    result_folder = result_folder+get_time_YYYYMMDDHH()+'_'+folder_description
    ensure_directory_exists(result_folder)

    branch2_dim = [2, 32, 32, 64]  # Source location branch
    trunk_dim = [2, 100, 100, 64]  # Trunk network (grid coordinates)

    geometry_dim = EXPECTED_IMG_SIZE  

    model = opnn(branch2_dim, trunk_dim, geometry_dim).to(device) # for CNN
    total_params = sum(p.numel() for p in model.parameters())
    print("total parameters")
    print(total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('TOTAL TRAINABLE PARAMS')
    print(trainable_params)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)  # Reduce LR every 500 steps

    # Prepare data
    dataloader_train, dataloader_valid,dataloader_test = load_data_by_split(DATA_PATH, bz)

    # train the model
    print('-'*15, 'TRAIN', '-'*15)
    trainer = Trainer(model, optimizer, device, num_epochs)
    trainer.set_result_path(result_folder)
    model = trainer.train(dataloader_train, dataloader_valid, dataloader_test, scheduler)

    # plot Losses
    file_paths = [trainer.train_log_path,trainer.val_log_path]
    plot_logs(file_paths, output_image=os.path.join(trainer.result_folder, "loss_plot.png"))

    
if __name__ == "__main__":
    print(f"CONFIG: DATA_PATH: {DATA_PATH}, RESULT_FOLDER: {RESULT_FOLDER}, epochs: {epochs}, VIZ_epoch_period: {VIZ_epoch_period}, BATCHSIZE: {BATCHSIZE}, STEP_SIZE: {STEP_SIZE}, EXPECTED_IMG_SIZE: {EXPECTED_IMG_SIZE}, EXPECTED_SIM_SIZE: {EXPECTED_SIM_SIZE}")
    parser = argparse.ArgumentParser(description="Experiment id or brief description, no space or slash allowed. Good Example: high_resolution_1.")
    parser.add_argument('exp_description', type=str, nargs='?', default="", help='Optional Experiment description. Good Example: high_resolution_1.')
    args = parser.parse_args()

    main(bz=BATCHSIZE,num_epochs=epochs, result_folder=RESULT_FOLDER, folder_description=args.exp_description)
    