import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import math
from math import log

def img_load(path, gray=False):
    """ Load batched tensor image (1,C,H,W) from file path.
    """
    if gray:
        return to_tensor(Image.open(path).convert('L'))[None,...]
    return to_tensor(Image.open(path))[None,...]

def awgn(input, noise_std, dist = 'uniform', log_base = 10, eps = 1e-8):
    """ Additive White Gaussian Noise
    y: clean input image
    noise_std: (tuple) noise_std of batch size N is uniformly sampled 
                between noise_std[0] and noise_std[1]. Expected to be in interval
                [0,1]
    """

    if not isinstance(noise_std, (list, tuple)):
        sigma = noise_std
    elif isinstance(noise_std, (list, tuple)) and dist == 'uniform': # uniform sampling of sigma
        sigma = noise_std[0] + \
               (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
    elif isinstance(noise_std, (list, tuple)) and dist == 'log':
        # Draw uniform on [log(a), log(b)] and then exponentiate
        # log base controls the "temperature"
        sigma = log_base**(math.log(noise_std[0]+eps, log_base) + \
            (math.log10(noise_std[1], log_base) - math.log10(noise_std[0]+eps, log_base))*torch.rand(len(input),1,1,1, device=input.device))

    elif isinstance(noise_std, (list, tuple)) and dist == 'cosine':
        # \sigma = ((cos(X)+1)/2)^2, X ~ U([cos^-1(2*a^0.5-1), cos^-1(2*b^0.5-1)])
        # Draw uniform number [0, 1], map to our desired data range
        
        # The lower bound actually comes from b, not a, since arccos is monotonically decreasing

        x = math.acos(2*noise_std[1]**0.5-1) + \
        (math.acos(2*noise_std[0]**0.5-1)-math.acos(2*noise_std[1]**0.5-1))*torch.rand(len(input),1,1,1, device=input.device)

        sigma = ((torch.cos(x)+1)/2)**2
    return input + torch.randn_like(input) * (sigma), sigma


