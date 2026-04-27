import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from model_utils import uball_project, pre_process, pre_process_pair, post_process, power_method
from mri_utils import batched_mri_encoding, batched_mri_decoding, quant_complex
from functools import partial

def ST(x, t):
    ''' 
    Define an element wise soft thresholding operator on x with threshold t
    
    ST(x, t) = sgn(x)*ReLU(|x|-t)
    '''

    # Change sign to sgn to handle complex valued inputs
    return x.sgn()*nn.functional.relu(x.abs() - t)

def CLIP(z, t):
    '''
    Define an elementwise clipping operation on x with threshold t

    CLIP(z, t) = sgn(x)*max(|z|, t)
    '''

    # return z.sgn()*torch.clamp(z.abs(), max=t)
    return z - ST(z, t)

class LearnablePolynomial(nn.Module):
    def __init__(self, coeffs):
        super().__init__()
        # Assume coeffs is a 1D tensor with N entries, corresponding to an order of N-1
        # Determine order from polynomial coeffs
        self.order = coeffs.numel()-1
        
        # Initialize coefficients randomly
        self.coeffs = nn.Parameter(coeffs.clone().detach())
        
    def forward(self, x):
        # Only works for x 1D. 
        # We can assume that x is 1x1x1x1, so we should flatten it
        x = x.reshape(-1)
        return torch.vander(x, N=self.order + 1, increasing=True) @ self.coeffs


class ComplexConvTranspose2d(nn.Module):
    def __init__(self,  M, 
                        C, 
                        P, 
                        stride = 1, 
                        bias=False):
        super(ComplexConvTranspose2d, self).__init__()
        self.M = M
        self.C = C
        self.P = P
        self.s = stride
        self.bias = bias
        self.padding=(P-1)//2
        self.output_padding = 1

        # Initialize two separate Conv2D transpose blocks, to operate an real and imag separately
        self.conv_real = nn.ConvTranspose2d(M, C, P, stride = stride, padding=self.padding, output_padding = self.output_padding, bias=False)
        self.conv_imag = nn.ConvTranspose2d(M, C, P, stride = stride, padding=self.padding, output_padding = self.output_padding, bias=False)
        
    def __call__(self, x):
        # Assume x is a complex valued input
        x_real = torch.real(x)
        x_imag = torch.imag(x)
        out_real_part1 = self.conv_real(x_real) # W_real * x_real
        out_real_part2 = self.conv_imag(x_imag) # W_imag * x_imag
        out_imag_part1 = self.conv_real(x_imag) # W_real * x_imag
        out_imag_part2 = self.conv_imag(x_real) # W_imag * x_real

        # Combine results to form the complex output
        out_real = out_real_part1 - out_real_part2
        out_imag = out_imag_part1 + out_imag_part2
        
        out = torch.complex(out_real, out_imag)
        return out
        

class CDLNet(nn.Module):
    def __init__(self,  K = 3, # number of CDL blocks
                        M = 64, # number of filters in each filter bank
                        P = 7, # kernel size
                        s = 1, # stride
                        C = 1, # number of input channels (either 1 or 3)
                        t0 = 0, # initial threshold
                        adaptive = False, # noise adaptive thresholds
                        init = True): # False -> use power method for weight init
        super(CDLNet, self).__init__()
        
        # Initialize A, B, D, t
        
        # A is a convolutional analysis operators (take channel dimension from 1 -> M or 3 -> M)
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride = s, padding=(P-1)//2, bias=False, dtype = torch.cfloat) for _ in range(K)])
        # B is a convolutional synthesis operator (take channel dimension from M -> 1 or M -> 3)
        self.B = nn.ModuleList([ComplexConvTranspose2d(M, C, P, stride = s, bias=False) for _ in range(K)])
        
        # D is the convolutional dictionary
        self.D = self.B[0] # alias D to B[0], otherwise unused as z0 is 0
        
        # t are the learned thresholds for the ST operator (K (num folds) x 2 (noise adaptive thresh) x M (num channels))
        self.t = nn.Parameter(t0*torch.ones(K, 2, M, 1, 1))
    
        # weight initialization (important! must initialize weights same weights for A and B)
        W = torch.randn(M, C, P, P, dtype = torch.cfloat)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            # Have to initialize real and imaginary parts of transposed conv separately
            # self.B[k].weight.data = W.conj().clone()
            self.B[k].conv_real.weight.data = torch.real(W).clone()
            self.B[k].conv_imag.weight.data = -1*torch.imag(W).clone()
    
        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                # power method returns the dominant eigenvalue for a matrix 
                L = power_method(DDt, torch.rand(1,C,128,128, dtype = torch.cfloat), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if np.real(L) < 0 or np.abs(np.imag(L)) > 1e-3:
                    print(f"STOP: something is very very wrong... {L:.1e}")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0]), i.e. divide by dominant singular value/sqrt of dominant eigenvalue
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(np.abs(L))
                self.B[k].conv_real.weight.data /= np.sqrt(np.abs(L))
                self.B[k].conv_imag.weight.data /= np.sqrt(np.abs(L))
                
        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.C = C
        self.t0 = t0
        self.adaptive = adaptive
        
        
        
    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds
        """
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            # self.B[k].weight.data = uball_project(self.B[k].weight.data)
            # Since the weight filters in our B's are separated into real and imag parts, we need to combine them and then uball project
            B_weights_complex = torch.complex(self.B[k].conv_real.weight.data, self.B[k].conv_imag.weight.data)
            B_weights_complex = uball_project(B_weights_complex)
            # Write them to our filters separately
            self.B[k].conv_real.weight.data = torch.real(B_weights_complex)
            self.B[k].conv_imag.weight.data = torch.imag(B_weights_complex)
        
    def forward(self, y, sigma = None, mask = 1):
        # mean subtraction and stride padding 
        yp, params = pre_process(y, self.s, mask = mask)
        
        # Threshold scale factor
        c = 0 if sigma is None or not self.adaptive else sigma
        # Perform K ISTA iterations
        # Initialize zp:
        # z_next = ST(z - A^T(Bz-y), thresh), but first z_0 = 0
        temp = self.A[0](yp)
        z = ST(temp, self.t[0,:1] + c*self.t[0,1:2])
        for k in range(self.K):
            temp = mask* self.B[k](z)
            # z = ST(z - self.A[k](temp-yp[:, :, 0:temp.shape[2], 0:temp.shape[3]]), self.t[k, 0:1] + c * self.t[k, 1:2])
            z = ST(z - self.A[k](temp-yp), self.t[k, 0:1]+c*self.t[k, 1:2])
        # x_hat = Dz
        x_hat = post_process(self.D(z), params)
        
        return x_hat, z
    
    def forward_generator(self, y, sigma=None, mask=1):
        """ same as forward but yields intermediate sparse codes
        """
        yp, params = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2]); yield z
        for k in range(1, self.K):
            z = ST(z - self.A[k](mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2]); yield z
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        yield xhat
        
class LPDSNet(nn.Module):
    def __init__(self,  K = 3, # number of LPDSNet blocks
                        M = 64, # number of filters in each filter bank
                        P = 7, # kernel size
                        s = 1, # stride
                        C = 1, # number of input channels (either 1 or 3)
                        l0 = 1e-3, # initial threshold
                        eta_0 = 0.5,
                        theta_0 = 0.,
                        poly_coeffs = torch.tensor([-1.1, 0, 0]), # sigmoid inverse of 0.25
                        adaptive = False, # noise adaptive threshold
                        e2e_diff = False, # Indicate whether or not we use this for diffusion, then we need a 2nd noise input
                        init = True): # False -> use power method for weight init
        super(LPDSNet, self).__init__()

        # Initialize A, B, D, t
        # A is a convolutional analysis operators (take channel dimension from 1 -> M or 3 -> M)
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride = s, padding=(P-1)//2, bias=False, dtype = torch.cfloat) for _ in range(K)])
        # B is a convolutional synthesis operator (take channel dimension from M -> 1 or M -> 3)
        self.B = nn.ModuleList([ComplexConvTranspose2d(M, C, P, stride = s, bias=False) for _ in range(K)])

        # D is the convolutional dictionary
        self.D = self.B[0] # alias D to B[0], otherwise unused as z0 is 0

        # t are the learned thresholds for the elementwise clipping (K (num folds) x 2 (noise adaptive thresh) x M (num channels))
        self.l = nn.Parameter(torch.cat((l0*torch.ones(K, 1, M, 1, 1), torch.zeros(K, 1, M, 1, 1)), dim = 1))
        
        # Set lambda_0 to l0, lambda_1 to 0.
        if e2e_diff:
            self.l_2 = nn.Parameter(torch.zeros(K, 1, M, 1, 1))
            # For our ImMAP2.5 architecture, we parameterize eta and beta as functions of sigma, learnable  polynomials. 
            self.eta = nn.ModuleList([LearnablePolynomial(coeffs = poly_coeffs) for k in range(K)])
            # We have a beta to control step size on (x_hat-x)
            self.beta = nn.ModuleList([LearnablePolynomial(coeffs = poly_coeffs) for k in range(K)])
            self.immap2p5 = True
        else:
            self.l_2 = torch.zeros(K, 1, M, 1, 1)
            # initialize eta and theta
            self.eta = nn.Parameter(eta_0 * torch.ones(K, 1))
            # Beta is not used in this case
            self.beta = torch.zeros_like(self.eta)
            self.immap2p5 = False
        # theta initialization is consistent
        self.theta = nn.Parameter(theta_0 * torch.ones(K, 1))
        # weight initialization (important! must initialize weights same weights for A and B)
        W = torch.randn(M, C, P, P, dtype = torch.cfloat)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            # Have to initialize real and imaginary parts of transposed conv separately
            self.B[k].conv_real.weight.data = torch.real(W).clone()
            self.B[k].conv_imag.weight.data = -1*torch.imag(W).clone()

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                # power method returns the dominant eigenvalue for a matrix
                L = power_method(DDt, torch.rand(1,C,128,128, dtype = torch.cfloat), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if np.real(L)  < 0 or np.imag(L) > 1e-3:
                    print(f"STOP: something is very very wrong... L = {L}")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0]), i.e. divide by dominant singular value/sqrt of dominant eigenvalue
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(np.abs(L))
                self.B[k].conv_real.weight.data /= np.sqrt(np.abs(L))
                self.B[k].conv_imag.weight.data /= np.sqrt(np.abs(L))

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.C = C
        self.l0 = l0
        self.adaptive = adaptive
    def forward(self, y, sigma=None, mask = None, smaps = None, mri = False):
        # y         B x C x H x W
        # smaps     B x C x H x W
        # mask      B x 1 x H x W
        if mri:
            E = partial(batched_mri_encoding, mask = mask, smaps = smaps)
            EH = partial(batched_mri_decoding, mask = mask, smaps = smaps)
        else:
            # If not MRI, then it's a denoising task
            E = nn.Identity()
            EH = nn.Identity()
        # Apply forward measurement operator 
        EHy = EH(y)
        # mean subtraction and stride padding 
        yp, params = pre_process(EHy, self.s, mask = 1)
        # Threshold scale factor
        c1 = 0 if sigma is None or not self.adaptive else sigma

        x_prev = torch.zeros_like(EHy)
        # x_prev = yp
        z = torch.zeros_like(self.A[0](x_prev))

        # Perform K-1 LDPS  iterations
        for k in range(0, self.K):
            x = x_prev - self.eta[k]*(EH(E(x_prev))-yp+self.B[k](z))
            xp = x + self.theta[k]*(x-x_prev)
            z = CLIP(z + self.A[k](xp), self.l[k, :1] + c1*self.l[k, 1:2])
            x_prev = x
    
        x_hat = post_process(x, params)
        return x_hat, z

    def forward_double_noise(self, y, sigma=None, mask = None, smaps = None, mri = False, x_init = None, sigma_t = None, ssdu_mask = None):
        # Will need to do this for MRI reconstruction
        # set up encoding/decoding operators
        # Flatten y, smaps
        # y         B x C x H x W
        # x_init    B x 1 x H x W
        # smaps     B x C x H x W
        # mask      B x 1 x H x W
        # ssdu_mask    B x 1 x H x W
        if mri is True:
            E = partial(batched_mri_encoding, mask = mask, smaps = smaps)
            EH = partial(batched_mri_decoding, mask = mask, smaps = smaps)
        else:
            # If not MRI, then it's a denoising task
            E = nn.Identity()
            EH = nn.Identity()
        # We can similarly consider corruption via random masks on the data fidelity step in the image domain. 
        if ssdu_mask is not None:
            # Since our x_t's are image domain, we actually don't need smaps. Instead of writing a new function, we can still used batched mri encoding but instead take smaps to be fully ones since the coil number is 1. 
            ssdu_smaps = torch.ones_like(y[:, 0, :, :]) # reduce channel dimension to 1, we consider the single coil case. 
            E_z = partial(batched_mri_encoding, mask = ssdu_mask, smaps = ssdu_smaps)
            E_zH = partial(batched_mri_decoding, mask = ssdu_mask, smaps = ssdu_smaps)
        else:
            E_z = nn.Identity()
            E_zH = nn.Identity()
        
        # Apply forward measurement operator 
        EHy = EH(y)
        # Since we want to use x_init in the iterations, we require that it be mean-subtracted. The best way to do so is to jointly normalize two EHy and x_init. 
        # mean subtraction and stride padding 
        yp, x_hat_p, params = pre_process_pair(EHy, x_init, self.s, mask = 1)
        # Threshold scale factor
        c1 = 0 if sigma is None or not self.adaptive else sigma
        # Diffusion scale factor
        c2 = 0 if sigma_t is None or not self.adaptive else sigma_t
        
        # If we do not want to initialize x as anything in particular, set it to 0
        '''       
        if x_init is None:
            x_prev = torch.zeros_like(EHy)
        else:
            x_prev = x_hat_p
        '''
        # With this new architecture, return back to the old x zero initialization
        x_prev = torch.zeros_like(EHy)
        z = torch.zeros_like(self.A[0](x_prev))
        
        # Perform K-1 LDPS  iterations
        for k in range(0, self.K):
            # If initialized correctly, we should have eta[k] and beta[k] be functions of c2
            eta_k = torch.sigmoid(self.eta[k](c2))
            beta_k = torch.sigmoid(self.beta[k](c2))
            # Do not force the step size on the B[k]'s to be parameterized in terms of sigma
            x = x_prev - eta_k*(EH(E(x_prev))-yp)-beta_k*E_zH(E_z(x_prev-x_hat_p))-0.25*self.B[k](z)
            xp = x + self.theta[k]*(x-x_prev)
            z = CLIP(z + self.A[k](xp), self.l[k, :1] + c1*self.l[k, 1:2] + c2*self.l_2[k])
            x_prev = x

        x_hat = post_process(x, params)
        return x_hat, z

    @torch.no_grad()
    def project(self):

        """ \ell_2 ball projection for filters, R_+ projection for thresholds, clip values, step sizes

        """
        self.l.clamp_(0.0)
        self.l_2.clamp_(0.0)
        # Actually not sure how to handle projection for eta and beta in the immap2p5 case. 
        if self.immap2p5 is False:
            self.eta.clamp_(0.0)
        self.theta.clamp_(min = 0.0, max = 1.0)
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            # self.B[k].weight.data = uball_project(self.B[k].weight.data)
            # Since the weight filters in our B's are separated into real and imag parts, we need to combine them and then uball project
            B_weights_complex = torch.complex(self.B[k].conv_real.weight.data, self.B[k].conv_imag.weight.data)
            B_weights_complex = uball_project(B_weights_complex)
            # Write them to our filters separately
            self.B[k].conv_real.weight.data = torch.real(B_weights_complex)
            self.B[k].conv_imag.weight.data = torch.imag(B_weights_complex)


# Create an LPDSNet-Base model so we don't mess up our MRI recon models. 
class LPDSNetBase(nn.Module):
    def __init__(self,  K = 3, # number of LPDSNet blocks
                        M = 64, # number of filters in each filter bank
                        P = 7, # kernel size
                        s = 1, # stride
                        C = 1, # number of input channels (either 1 or 3)
                        l0 = 1e-3, # initial threshold
                        eta_0 = 0.5,
                        theta_0 = 0.,
                        adaptive = False, # noise adaptive threshold
                        e2e_diff = False, # Indicate whether or not we use this for diffusion, then we need a 2nd noise input
                        init = True): # False -> use power method for weight init
        super(LPDSNetBase, self).__init__()

        # Initialize A, B, D, t
        # A is a convolutional analysis operators (take channel dimension from 1 -> M or 3 -> M)
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride = s, padding=(P-1)//2, bias=False, dtype = torch.cfloat) for _ in range(K)])
        # B is a convolutional synthesis operator (take channel dimension from M -> 1 or M -> 3)
        self.B = nn.ModuleList([ComplexConvTranspose2d(M, C, P, stride = s, bias=False) for _ in range(K)])

        # D is the convolutional dictionary
        self.D = self.B[0] # alias D to B[0], otherwise unused as z0 is 0

        # t are the learned thresholds for the elementwise clipping (K (num folds) x 2 (noise adaptive thresh) x M (num channels))
        self.l = nn.Parameter(torch.cat((l0*torch.ones(K, 1, M, 1, 1), torch.zeros(K, 1, M, 1, 1)), dim = 1))

        # Set lambda_0 to l0, lambda_1 to 0.
        if e2e_diff:
            self.l_2 = nn.Parameter(torch.zeros(K, 1, M, 1, 1))
        else:
            self.l_2 = torch.zeros(K, 1, M, 1, 1)
        # weight initialization (important! must initialize weights same weights for A and B)
        W = torch.randn(M, C, P, P, dtype = torch.cfloat)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            # Have to initialize real and imaginary parts of transposed conv separately
            self.B[k].conv_real.weight.data = torch.real(W).clone()
            self.B[k].conv_imag.weight.data = -1*torch.imag(W).clone()

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                # power method returns the dominant eigenvalue for a matrix
                L = power_method(DDt, torch.rand(1,C,128,128, dtype = torch.cfloat), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if np.abs(L)  < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0]), i.e. divide by dominant singular value/sqrt of dominant eigenvalue
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(np.abs(L))
                self.B[k].conv_real.weight.data /= np.sqrt(np.abs(L))
                self.B[k].conv_imag.weight.data /= np.sqrt(np.abs(L))

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.C = C
        self.l0 = l0
        self.adaptive = adaptive

        # initialize eta and theta
        self.eta = nn.Parameter(eta_0 * torch.ones(K, 1))
        self.theta = nn.Parameter(theta_0 * torch.ones(K, 1))

    def forward(self, y, sigma=None, mask = None, smaps = None, mri = False):
        # y         B x C x H x W
        # smaps     B x C x H x W
        # mask      B x 1 x H x W
        if mri:
            E = partial(batched_mri_encoding, mask = mask, smaps = smaps)
            EH = partial(batched_mri_decoding, mask = mask, smaps = smaps)
        else:
            # If not MRI, then it's a denoising task
            E = nn.Identity()
            EH = nn.Identity()
        # Apply forward measurement operator
        EHy = EH(y)
        # mean subtraction and stride padding
        yp, params = pre_process(EHy, self.s, mask = 1)
        # Threshold scale factor
        c1 = 0 if sigma is None or not self.adaptive else sigma

        x_prev = torch.zeros_like(EHy)
        z = torch.zeros_like(self.A[0](x_prev))

        # Perform K-1 LDPS  iterations
        for k in range(0, self.K):
            x = x_prev - self.eta[k]*(EH(E(x_prev))-yp+self.B[k](z))
            xp = x + self.theta[k]*(x-x_prev)
            z = CLIP(z + self.A[k](xp), self.l[k, :1] + c1*self.l[k, 1:2])
            x_prev = x

        x_hat = post_process(x, params)
        return x_hat, z
    @torch.no_grad()
    def project(self):

        """ \ell_2 ball projection for filters, R_+ projection for thresholds, clip values, step sizes

        """
        self.l.clamp_(0.0)
        self.l_2.clamp_(0.0)
        self.eta.clamp_(0.0)
        self.theta.clamp_(min = 0.0, max = 1.0)
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            # self.B[k].weight.data = uball_project(self.B[k].weight.data)
            # Since the weight filters in our B's are separated into real and imag parts, we need to combine them and then uball project
            B_weights_complex = torch.complex(self.B[k].conv_real.weight.data, self.B[k].conv_imag.weight.data)
            B_weights_complex = uball_project(B_weights_complex)
            # Write them to our filters separately
            self.B[k].conv_real.weight.data = torch.real(B_weights_complex)
            self.B[k].conv_imag.weight.data = torch.imag(B_weights_complex)
