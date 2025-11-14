import torch
from tqdm import tqdm
import numpy as np

def conj_grad(A, b, x0 = None, tol = 1e-6, max_iter = 100):
    '''
    Let A be a Pytorch operator, necessarily symmetric, positive semi-definite
    We solve the system Ax = b
    '''
    if x0 is None:
        x0 = torch.zeros_like(b)
    # Compute first residual
    r = b - A(x0)
    p = r
    x = x0
    tol_reached = False
    for k in tqdm(range(int(max_iter))):
        # Apply operator to p
        Ap = A(p)
        # Implement inner products as elementwise sums
        alpha = torch.sum(r.conj() * r) / (torch.sum(p.conj() * Ap) + 1e-8) 
        x = x + alpha * p
        r_next = r - alpha * Ap
        r_norm = torch.norm(r, 2)
        if r_norm <= tol:
            tol_reached = True
            return x, tol_reached
        beta = torch.sum(r_next.conj() *  r_next)/(torch.sum(r.conj() *  r) + 1e-8)
        p = r_next + beta * p
        r = r_next
        breakpoint()
        print(f"Iteration: {k}, Residual: {r_norm}")
    return x, tol_reached
