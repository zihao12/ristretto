"""
Nonnegative Matrix Factorization With GKL.
"""

from __future__ import division

import numpy as np
from scipy import linalg

from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast
from sklearn.decomposition.nmf import _initialize_nmf
from sklearn.utils import check_random_state

from .qb import compute_rqb
from .nmf import compute_rnmf

import sys
import time

_VALID_DTYPES = (np.float32, np.float64)



def update_kl(A, W, H,E, eps = 0):
    """ 
    W = W .* (((A ./ (W*H)) * H.') ./ (E * H.'));
    W = max(W,eps);
    H = H .* ((W.' * (A ./ (W*H))) ./ (W.' * E));
    H = max(H,eps);
    """
    mulW = (A/W.dot(H)).dot(H.T)/(E.dot(H.T))
    W *= mulW
    W = W.clip(min = eps)

    mulH = W.T.dot(A/W.dot(H))/(W.T.dot(E))
    H *= mulH
    H = H.clip(min = eps)

    return W, H

def cost(A, W,H, eps = 0):
    """
    sum(sum(WH - A.*log(WH + eps)))
    """
    WH = W.dot(H)
    return (WH - A *np.log(WH + eps)).sum()




def compute_nmf_kl(A, rank, init='nndsvda', eps = sys.float_info.min, shuffle=False,
                l2_reg_H=0.0, l2_reg_W=0.0, l1_reg_H=0.0, l1_reg_W=0.0,
                tol=1e-5, maxiter=200, random_state=None):
   
    random_state = check_random_state(random_state)

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if np.any(A < 0):
        raise ValueError("Input matrix with nonnegative elements is required.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    W, H = _initialize_nmf(A, rank, init=init, eps=1e-6, random_state=random_state)

    costs = []
    E = np.ones(A.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the mu algorithm until maxiter is reached
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for niter in range(maxiter):

        W, H = update_kl(A, W, H,E, eps = eps)
    # Return factor matrices
    return W, H


def compute_rnmf_kl(A, rank, oversample=100, init='nndsvda',eps = sys.float_info.min,
                 tol=1e-5, maxiter=200, random_state=None, approx = 'nndsvd'):
    
    random_state = check_random_state(random_state)

    # converts A to array, raise ValueError if A has inf or nan
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    flipped = False
    if n > m:
        A = A.T
        m, n = A.shape
        flipped = True

    # if A.dtype not in _VALID_DTYPES:
    #     raise ValueError('A.dtype must be one of %s, not %s'
    #                      % (' '.join(_VALID_DTYPES), A.dtype))

    if np.any(A < 0):
        raise ValueError("Input matrix with nonnegative elements is required.")


    
    # compute low rank "projection"
    # I hope to get A \approx Q' * B, where 
    # Q (p,d) is orthonormal, nonnegative
    # B (d,n) is nonnegative

    ## one way: just use nndsvd
    if approx == 'nndsvd':
        start = time.time()
        Q, B = _initialize_nmf(A, rank+oversample, init="nndsvd", eps=1e-6, random_state=random_state)
        print("approximation takes {}".format(time.time() - start))

    ## the other way: use rnmf
    if approx == 'rnmf':
        start = time.time()
        Q, B = compute_rnmf(A, rank+oversample, init = "nndsvd")
        print("approximation takes {}".format(time.time() - start))

    #  Initialization methods for factor matrices W and H
    W, H = _initialize_nmf(A, rank, init=init, eps=1e-6, random_state=random_state)
    Ht = np.array(H.T, order='C')
    W_tilde = Q.T.dot(W)
    del A

    costs = []
    E = np.ones(B.shape)

    #  Iterate the mu algorithm until maxiter is reached
    for niter in range(maxiter):

        W_tilde, H = update_kl(B, W_tilde, H,E, eps = eps)

        W = Q.dot(W_tilde)
        W = W.clip(min = eps)
        W_tilde = Q.T.dot(W)

    # Return factor matrices
    if flipped:
        return(Ht, W.T)
    return W, H
