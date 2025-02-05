import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(cuda_device=None):
    if cuda_device:
        return torch.device(cuda_device)
    else:
        return torch.device('cpu')

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
