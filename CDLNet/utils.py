import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

def img_load(path, gray=False):
	""" Load batched tensor image (1,C,H,W) from file path.
	"""
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

def gen_bayer_mask(x):
    ''' 
    RGB --> [ R  G ]
            [ G  B ]
    '''    
    m = torch.zeros_like(x)
    m[:,0,0::2,0::2] = 1 # R
    m[:,1,0::2,1::2] = 1 # G1
    m[:,1,1::2,0::2] = 1 # G2
    m[:,2,1::2,1::2] = 1 # B
    return m

def awgn(input, noise_std):
	""" Additive White Gaussian Noise
	y: clean input image
	noise_std: (tuple) noise_std of batch size N is uniformly sampled 
	           between noise_std[0] and noise_std[1]. Expected to be in interval
			   [0,255]
	"""
	if not isinstance(noise_std, (list, tuple)):
		sigma = noise_std
	else: # uniform sampling of sigma
		sigma = noise_std[0] + \
		       (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
	return input + torch.randn_like(input) * (sigma/255), sigma

