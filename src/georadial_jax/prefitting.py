import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner 
from georadial import data_gridding
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, rosen, rosen_der
from scipy.special import j0, j1, jn_zeros, jv
import os 
import glob
import random

ARCSEC_TO_RAD= 1/206265.0


def chi_for_positional_off(theta, d_data, u_d, v_d ):
    delta_x = theta[0] 
    delta_y = theta[1]
    diag_mat_cos_inv = np.cos(2 * np.pi * (delta_x * u_d + delta_y * v_d))
    diag_mat_sin_inv = np.sin(2 * np.pi * (delta_x * u_d + delta_y * v_d))

    n_d = int(len(d_data)/2)
    d_real = d_data[:n_d]
    d_imag = d_data[n_d:]
    d_imag_mod = + d_real * diag_mat_sin_inv + d_imag * diag_mat_cos_inv
    return np.sum(d_imag_mod**2)




def determine_positional_offsets(d_data, u_d, v_d):
    res = minimize(chi_for_positional_off, [0,0], method='BFGS',\
        args=(d_data, u_d, v_d), bounds =( (-ARCSEC_TO_RAD, ARCSEC_TO_RAD), (-ARCSEC_TO_RAD, ARCSEC_TO_RAD)))
    return res.x