import numpy as np
from scipy.fft import fft
import torch
from torch import nn

def preprocess(x, max_width=50000):
    x_padded = zero_padding(x, max_width=max_width)
    x_am_fft = fast_fourier_transform(x_padded)
    x_input = amplitude_normalization(x_am_fft)

    return x_input

def zero_padding(x, max_width):
    # Differing sizes of signals
    if isinstance(x, list):
        x_padded = []

        for x_ in x:
            if not isinstance(x, torch.Tensor):
                x_ = torch.tensor(x_)

            pad_width = max_width - x_.shape[-1]
            m = nn.ConstantPad1d((0, pad_width), 0)

            x_padded.append(m(x_).numpy())
        
        x_padded = np.array(x_padded)

    else:
    # Singular entry or all same size
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        pad_width = max_width - x.shape[-1]
        m = nn.ConstantPad1d((0, pad_width), 0)
    
        x_padded = m(x).numpy()

    return x_padded

def fast_fourier_transform(x):
    x_fft = fft(x)
    x_fft_conj = np.conjugate(x_fft)
    x_am_fft = np.sqrt(np.multiply(x_fft, x_fft_conj))
    
    return np.real(x_am_fft)

def amplitude_normalization(x):
    return x/np.max(x, axis=-1)