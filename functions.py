import pandas as pd
import os.path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Dataset(Dataset):
    def __init__(self, stim, obj):
        self.features = stim
        self.target = obj

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.target[idx]

        return X, y
    
def save_activations(activations,data_type, rnd = False):
    
    for layer_name in activations.keys() :
        act_to_store = pd.DataFrame(activations[layer_name])
        
        if rnd == False : path_to_store = f'Activations/{data_type}_{layer_name}.csv'
        else : path_to_store = f'Activations_RND/{data_type}_{layer_name}.csv'
    
        act_to_store.to_csv(path_to_store)
    
        print(f'saving complete in {path_to_store}')
        
        
def load_activations(data_type, rnd = False):
    if rnd == True : layers = [ 'layer2']
    else : layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    ACTIVATIONS = {}
    for layer_name in layers :
        if rnd == False : path = f'Activations/{data_type}_{layer_name}.csv'
        else : path = f'Activations_RND/{data_type}_{layer_name}.csv'
    
        act_df = pd.read_csv(path)
    
        # Size = [., 1001] need to delete the first column for index
        act_df = act_df.drop('Unnamed: 0', axis=1)
    
        ACTIVATIONS[layer_name] = act_df.to_numpy()
    
    return ACTIVATIONS