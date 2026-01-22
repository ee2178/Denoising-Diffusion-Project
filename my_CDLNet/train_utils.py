import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from math import log

def img_load(path, gray=False):
    """ Load batched tensor image (1,C,H,W) from file path.
    """
    if gray:
        return to_tensor(Image.open(path).convert('L'))[None,...]
    return to_tensor(Image.open(path))[None,...]

def awgn(input, noise_std, dist = 'uniform'):
    """ Additive White Gaussian Noise
    y: clean input image
    noise_std: (tuple) noise_std of batch size N is uniformly sampled 
                between noise_std[0] and noise_std[1]. Expected to be in interval
                [0,255]
    """

    if not isinstance(noise_std, (list, tuple)):
        sigma = noise_std
    elif isinstance(noise_std, (list, tuple)) and dist == 'uniform': # uniform sampling of sigma
        sigma = noise_std[0] + \
               (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
    elif isinstance(noise_std, (list, tuple)) and dist == 'log-uniform':
        # log uniform has support [a, b], corresponding to uniform w/ support [exp(a), exp(b)].
        # We can usually assume a = 0, b = 1 for diffusion training
        # So, draw a uniform sample from [exp(a), exp(b)], then take its log
        # Insert tiny epsilon so to not run into numerical stability issues
        eps = 1e-8
        sigma = torch.exp((log(noise_std[0]+eps) + \
            (log(noise_std[1]) - log(noise_std[0]+eps))*torch.rand(len(input),1,1,1, device=input.device)))
        
    return input + torch.randn_like(input) * (sigma/255), sigma


