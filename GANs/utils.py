import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_trainable_gradients(model, flag):
    """
    Sets the parameters of the given model to trainable or not,
    depending on the set flag
    """
    for p in model.parameters:
        p.requires_grad = flag

def sample_noise(batch_size, dim=100, device=device):
    """
    Samples N d-dimensional noise vectors that are randomly distributed 
    according to a Gaussian distribution, where N is the batch size 
    """
    z = torch.FloatTensor(
        batch_size, dim
    ).to(device)
    z.data.normal_() # Ensure normally distributed elements

    return z
