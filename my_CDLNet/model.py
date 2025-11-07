import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from model_utils import uball_project, pre_process, post_process, power_method

def ST(x, t):
    ''' 
    Define an element wise soft thresholding operator on x with threshold t
    
    ST(x, t) = sgn(x)*ReLU(|x|-t)
    '''

    # Change sign to sgn to handle complex valued inputs
    return x.sgn()*nn.functional.relu(x.abs() - t)

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
        
        # Initialize two separate Conv2D transpose blocks, to operate an real and imag separately
        self.conv_real = nn.ConvTranspose2d(M, C, P, stride = stride, padding=self.padding, bias=False, dtype = torch.float64)
        self.conv_imag = nn.ConvTranspose2d(M, C, P, stride = stride, padding=self.padding, bias=False, dtype = torch.float64)
        
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
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        # Perform K ISTA iterations
        # Initialize zp:
        # z_next = ST(z - A^T(Bz-y), thresh), but first z_0 = 0
        temp = self.A[0](yp)
        z = ST(temp, self.t[0,:1] + c*self.t[0,1:2])
        for k in range(self.K):
            temp = mask* self.B[k](z)
            z = ST(z - self.A[k](temp-yp[:, :, 0:temp.shape[2], 0:temp.shape[3]]), self.t[k, 0:1] + c * self.t[k, 1:2])
        
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
        


# def ST(x, t):
#     ''' 
#     Define an element wise soft thresholding operator on x with threshold t
    
#     ST(x, t) = sign(x)*ReLU(|x|-t)
#     '''
#     return x.sign()*F.relu(x.abs() - t)

# # Assume no masking (we are not interested in demosaicing, just denoising). Assume we always want noise adaptive thresholds
# class CDLNet(nn.Module):
#     def __init__(self,  K = 3, # number of CDL blocks
#                         M = 64, # number of filters in each filter bank
#                         P = 7, # kernel size
#                         s = 1, # stride
#                         C = 1, # number of input channels (either 1 or 3)
#                         t0 = 0, # initial threshold,
#                         adaptive = True, 
#                         init = True): 
#         super(CDLNet, self).__init__()
        
#         # Initialize A, B, D, t
        
#         # A is a convolutional analysis operators (take channel dimension from 1 -> M or 3 -> M)
#         self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride = s, padding=(P-1)//2, bias=False) for _ in range(K)])
#         # B is a convolutional synthesis operator (take channel dimension from M -> 1 or M -> 3)
#         self.B = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride = s, padding=(P-1)//2, bias=False) for _ in range(K)])
        
#         # D is the convolutional dictionary
#         self.D = self.B[0] # alias D to B[0], otherwise unused as z0 is 0
        
#         # t are the learned thresholds for the ST operator (K (num folds) x 2 (noise adaptive thresh) x M (num channels))
#         self.t = nn.Parameter(t0*torch.ones(K, 2, M, 1, 1))
#         W = torch.randn(M, C, P, P)
#         for k in range(K):
#             self.A[k].weight.data = W.clone()
#             self.B[k].weight.data = W.clone()
#         # Loading from a state dictionary
#         if init:
#             print("Running power-method on initial dictionary...")
#             with torch.no_grad():
#                 DDt = lambda x: self.D(self.A[0](x))
#                 # power method returns the dominant eigenvalue for a matrix 
#                 L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
#                 print(f"Done. L={L:.3e}.")

#                 if L < 0:
#                     print("STOP: something is very very wrong...")
#                     sys.exit()
    
#         # weight initialization (important! must initialize weights same weights for A and B)
        
                
#         # set parameters
#         self.K = K
#         self.M = M
#         self.P = P
#         self.s = s
#         self.C = C
#         self.t0 = t0
        
        
        
#     @torch.no_grad()
#     def project(self):
#         """ \ell_2 ball projection for filters, R_+ projection for thresholds
#         """
#         self.t.clamp_(0.0) 
#         for k in range(self.K):
#             self.A[k].weight.data = uball_project(self.A[k].weight.data)
#             self.B[k].weight.data = uball_project(self.B[k].weight.data)
        
#     def forward(self, y, sigma = None, mask = 1):
#         # mean subtraction and stride padding 
#         yp, params = pre_process(y, self.s)
        
#         # Threshold scale factor
#         c = sigma/255.0
        
#         # Perform K ISTA iterations
#         # Initialize zp:
#         # z_next = ST(z - A^T(Bz-y), thresh), but first z_0 = 0
#         z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2])
#         for k in range(self.K):
#             z = ST(z - self.A[k](mask*self.B[k](z)-yp), self.t[k, 0:1] + c * self.t[k, 1:2])
        
#         # x_hat = Dz
#         # Recenter at image mean 
#         x_hat = post_process(self.D(z), params)
        
#         return x_hat, z
