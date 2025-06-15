from pickle import GLOBAL
import torch
import os

###Global variables###
class GlobalParams:
    def __init__(self):
        self.batchSize = 32
        self.inputDim = 784
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GLOBAL_INFO = GlobalParams()