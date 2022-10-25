import functools
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader


class EcogDataset(data.Dataset):
    # Dataset for Ecog 
    def __init__(self, config, data):
        self.data = np.load(data)
        self.config = config
        
    def __getitem__(self, index):
        ecog = torch.from_numpy(self.data[index]).float()
        #ecog = torch.unsqueeze(ecog, 0)
        label = 0 # only one class
        return ecog, label
    
    def __len__(self):
        return self.data.shape[0]

def get_dataloader(config):
    print('Loading EcoG dataset')
    dataset = EcogDataset(config, 'data/data.npy')
    loader = functools.partial(
        DataLoader,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
    )
    loader_tr = loader(dataset = dataset, shuffle=True)
    print(f'Number of training samples: {len(dataset)}')
    print(f'Batch size: {config.batch_size}')
    return loader_tr

if __name__ == '__main__':
    from get_config import get_config
    config = get_config()
    dataloader = get_dataloader(config)