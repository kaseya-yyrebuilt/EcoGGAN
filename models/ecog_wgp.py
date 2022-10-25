import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn

def weights_init(m):
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.nz = config.nz
        self.net = nn.Sequential()
        self.outc_list = config.outc_list
        self.kern_list = config.kern_list
        self.strd_list = config.strd_list
        self.padd_list = config.padd_list
        inc = self.nz
        
        for i, outc in enumerate(self.outc_list):
            self.net.add_module(f'Conv1D-{i}', 
                                nn.ConvTranspose1d(inc, outc, self.kern_list[i], self.strd_list[i], self.padd_list[i],
                                                    bias=False)
                                )
            self.net.add_module(f'BN-{i}',
                                nn.BatchNorm1d(outc)
                                )
            self.net.add_module(f'ReLU-{i}',
                                nn.ReLU(True)
                                )
            inc = outc

        self.net.add_module('Conv1D-last',
                            nn.ConvTranspose1d(inc, config.inc, 4, 2, 1, bias=False)
                            )
        self.net.add_module('Tanh',
                            nn.Tanh()
                            )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
class Discriminator(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential()
        self.outc_list = config.outc_list[::-1]
        self.kern_list = config.kern_list[::-1]
        self.strd_list = config.strd_list[::-1]
        self.padd_list = config.padd_list[::-1]
        self.slope = config.lrl_slope
        inc = config.inc
        for i in range(len(self.outc_list)):
            self.net.add_module(f'Conv1d-{i}',
                                nn.Conv1d(inc, self.outc_list[i], self.kern_list[0], self.strd_list[0], self.padd_list[0],
                                          bias=False)
                                )
            if i > 0:
                self.net.add_module(f'BN-{i}',
                                    nn.BatchNorm1d(self.outc_list[i])
                                    )
            self.net.add_module(f'ReLU-{i}',
                                nn.LeakyReLU(self.slope, inplace=True)
                                )
            inc = self.outc_list[i]
            
        self.net.add_module('Conv1d-last',
                            nn.Conv1d(inc, 1, config.ks, 1, 0, bias=False)
                            )
        
    def forward(self, x, y=None):
        x = self.net(x)
        return x
        #return torch.sigmoid(x)
    
    
