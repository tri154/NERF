import numpy as np
from torch import nn
import torch as tc

devcie = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

# Model config:
class Config:
    def __init__(self):
        self.n_low_layers = 8
        self.skip_coarse = [4]
        self.coarse_d_filter = 

        
class NERF(nn.Module):
    def __init__(self,):
        super().__init__()
        cfg = Config()
        self.cfg = cfg
        def create_layer(n_in, n_out): return nn.Linear(n_in, n_out), nn.ReLU()
        layers = [create_layer(3, cfg.coarse_d_filter)]
        for i in range(cfg.n_low_layers - 1):
            if i in cfg.skip_coarse:
                layers += create_layer(3 + cfg.coarse_d_filter, cfg.coarse_d_filter)
            else:
                layers += create_layer(cfg.coarse_d_filter, cfg.coarse_d_filter)
        self.low_layers = nn.Sequential(layers)


        



        
