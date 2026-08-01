import torch
import torch as ch
import numpy as np

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def is_PD(x):
    return np.all(np.linalg.eigvals(x) > 0)

def get_nearest_PD(A, nit=10):
    '''
    Note from David Bau - I suspect this function is doing the wrong thing
    and finding the nearest PD *correlation* matrix, whereas what we want
    is the nearest PD (unnormalized, covariance) matrix.

    I have put the nearestPdCholesky function that does that, below.
    '''
    n = A.shape[0]
    W = np.identity(n) 
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

        
def zca_from_cov(cov, relative_floor=1e-6):
    """Build a stable ZCA whitening matrix from a second moment."""
    dtype = cov.dtype
    cov64 = cov.double()
    cov64 = 0.5 * (cov64 + cov64.t())
    evals, evecs = torch.linalg.eigh(cov64, UPLO='U')
    floor = evals.max().clamp_min(torch.finfo(evals.dtype).eps) * relative_floor
    inv_sqrt = evals.clamp_min(floor).rsqrt()
    return ((evecs * inv_sqrt.unsqueeze(0)) @ evecs.t()).to(dtype)


def zca_whitened_query_key(matrix, k):
    compute_dtype = torch.float32
    matrix = matrix.to(device=k.device, dtype=compute_dtype)
    k_float = k.to(compute_dtype)
    if len(k.shape) == 1:
        return torch.mm(matrix, k_float[:, None])[:, 0]
    return torch.mm(matrix, k_float.permute(1, 0)).permute(1, 0)


def zca_unwhitened_direction(matrix, direction):
    """Map whitened-coordinate directions back to activation space."""
    is_vector = direction.dim() == 1
    rhs = direction[:, None] if is_vector else direction.t()
    mapped = torch.linalg.solve(matrix.double(), rhs.double()).to(direction.dtype)
    return mapped[:, 0] if is_vector else mapped.t()
