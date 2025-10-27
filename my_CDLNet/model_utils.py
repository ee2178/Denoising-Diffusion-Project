import torch
import torch.nn.functional as F
import numpy as np

def power_method(A, b, num_iter=1000, tol=1e-6, verbose=True):
	"""
	Power method for pytorch operator A and initial vector b.
	"""
	eig_old = torch.zeros(1)
	flag_tol_reached = False
	for it in range(num_iter):
		b = A(b)
		b = b / torch.norm(b)
		eig_max = torch.sum(b.conj()*A(b))
		if verbose:
			print('i:{0:3d} \t |e_new - e_old|:{1:2.2e}'.format(it,abs(eig_max-eig_old).item()))
		if abs(eig_max-eig_old)<tol:
			flag_tol_reached = True
			break
		eig_old = eig_max
	if verbose:
		print('tolerance reached!',it)
		print(f"L = {eig_max.item():.3e}")
	return eig_max.item(), b, flag_tol_reached

def uball_project(W, dim=(2,3)):
    """ projection of W onto the unit ball
    """
    
    # Spectral norm along (2, 3) dimension
    normW = torch.linalg.norm(W, dim=dim, keepdim=True)
    return W * torch.clamp(1/normW, max=1)


def pre_process(x, stride, mask=1):
    """ image preprocessing: stride-padding and mean subtraction.
    """
    params = []
    # mean-subtract
    if torch.is_tensor(mask):
        xmean = x.sum(dim=(1,2,3), keepdim=True) / mask.sum(dim=(1,2,3), keepdim=True)
    else:
        xmean = x.mean(dim=(1,2,3), keepdim=True)
    x = mask*(x - xmean)
    params.append(xmean)
    # pad signal for stride
    pad = calc_pad_2D(*x.shape[2:], stride)
    x = F.pad(x, pad, mode='reflect')
    if torch.is_tensor(mask):
        mask = F.pad(mask, pad, mode='reflect')
    params.append(pad)
    return x, params

def post_process(x, params):
    """ undoes image pre-processing given params
    """
    # unpad
    pad = params.pop()
    x = unpad(x, pad)
    # add mean
    xmean = params.pop()
    x = x + xmean
    return x

def calc_pad_1D(L, M):
    """ Return pad sizes for length L 1D signal to be divided by M
    """
    if L%M == 0:
        Lpad = [0,0]
    else:
        Lprime = np.ceil(L/M) * M
        Ldiff  = Lprime - L
        Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
    return Lpad

def calc_pad_2D(H, W, M):
    """ Return pad sizes for image (H,W) to be divided by size M
    (H,W): input height, width
    output: (padding_left, padding_right, padding_top, padding_bottom)
    """
    return (*calc_pad_1D(W,M), *calc_pad_1D(H,M))

def conv_pad(x, ks, mode):
    """ Pad a signal for same-sized convolution
    """
    pad = (int(np.floor((ks-1)/2)), int(np.ceil((ks-1)/2)))
    return F.pad(x, (*pad, *pad), mode=mode)

def unpad(I, pad):
    """ Remove padding from 2D signalstack"""
    if pad[3] == 0 and pad[1] > 0:
        return I[..., pad[2]:, pad[0]:-pad[1]]
    elif pad[3] > 0 and pad[1] == 0:
        return I[..., pad[2]:-pad[3], pad[0]:]
    elif pad[3] == 0 and pad[1] == 0:
        return I[..., pad[2]:, pad[0]:]
    else:
        return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]
