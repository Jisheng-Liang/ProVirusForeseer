import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import copy, random
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def test_prediction(net, device, embed):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """

    with torch.no_grad():
        embed = embed.to(device)

        prediction= net(embed)
        prediction = prediction.squeeze(-1)
        prediction = prediction.detach().cpu().numpy()

    return prediction

class ProteinRegressor(nn.Module):
    def __init__(self):
        super(ProteinRegressor, self).__init__()

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        self.hidden_size = 46848
        # Classification layer
        self.hidden_layer1 = nn.Linear(self.hidden_size, 16384)
        self.hidden_layer2 = nn.Linear(16384, 4096)
        self.rgs_layer = nn.Linear(4096,1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
    @autocast()
    def forward(self, embed):

        output = self.hidden_layer1(embed); output = self.act(output); output = self.dropout(output)
        output = self.hidden_layer2(output); output = self.act(output); output = self.dropout(output)
        output = self.rgs_layer(output)

        return output

def main():
    # parameters
    set_seed(1)

    # dataset 
    print("Reading test data...")
    path_to_pt = "3.pt"
    embed = torch.load(path_to_pt)
    embed = embed.detach()
    # Creating instances of training and validation dataloaders
    # train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
    # val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)
    path_to_model = 'models/proteinLM_val_loss_0.06691_ep_9.pt'


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ProteinRegressor()
    # model.to(device)
    # if torch.cuda.device_count() > 1:  # if multiple GPUs
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    # model.to(device)
    
    prediction = test_prediction(net=model, device=device, embed=embed)  # set the with_labels parameter to False if your want to get predictions on a dataset without labels)
    print(prediction)

if __name__ == "__main__":
    main()