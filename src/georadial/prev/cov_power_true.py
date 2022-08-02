import scipy.special
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0, j1, jn_zeros, jv
import numpy as np
from scipy.stats import multivariate_normal
import emcee
import astropy.io.fits as pyfits
from georadial import data_load
from multiprocessing import Pool
import random
from scipy.linalg import block_diag
import os 
import pickle






def TSV_mat(nsize):
    tsv_mat = np.zeros((nsize, nsize))
    for i in range(nsize-1):
        tsv_mat[i][i] = 2
        tsv_mat[i+1][i] = -1
        tsv_mat[i][i+1] = -1
    tsv_mat[nsize-1][nsize-1] = 2
    return tsv_mat



def inv_RBF_for_vis(distance, alpha, gamma, simga = 1e-9):
    nx, ny = np.shape(distance)
    K=alpha * (np.exp(-(distance/gamma)**2/2) + simga * np.identity(nx))
    return K

def gauss_model_for_log(x, amp, sigma, x0):
    return amp * exp (-(x-x0)**2/sigma**2)


def cov_power(q_n, alpha, gamma, H_mat, q0=10**4):
    """Compute precision matrix for visibility considering powerlaw typye power spectrum

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        alpha (float): value determines the height of power spectrun at q0 (minimum visibility distance)
        gamma (float): power law index (q_k/q_0)**(-gamma)
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        K_cov_inv (ndarray): 2d array for precision matrix for visibility
        logdet_cov (float): determinant of covariance matrix
    """

    power_qk = alpha * (q_n/q0)**(-gamma)
    power_diag = np.diag(1/power_qk**2)
    K_cov_inv = H_mat.T@power_diag@H_mat
    (sign, logdet_inv_cov) = np.linalg.slogdet(K_cov_inv)
    logdet_cov = 1/logdet_inv_cov
    return K_cov_inv, logdet_cov

def cov_power_plus_gp(q_n, q_dist, alpha, gamma, alpha2, gamma2, H_mat, q0 = 10000.0):
    """Compute precision matrix for visibility considering powerlaw typye power spectrum

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        alpha (float): value determines the height of power spectrun at q0 (minimum visibility distance)
        gamma (float): power law index (q_k/q_0)**(-gamma)
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        K_cov_inv (ndarray): 2d array for precision matrix for visibility
        logdet_cov (float): determinant of covariance matrix
    """

    power_qk = alpha * (q_n/q0)**(-gamma)
    power_diag = np.diag(1/power_qk**2)
    K_vis = alpha2 * np.exp(-(q_dist/gamma2)**2/2)
    K_cov_inv = H_mat.T@(power_diag+np.linalg.inv(K_vis))@H_mat
    (sign, logdet_inv_cov) = np.linalg.slogdet(K_cov_inv)
    logdet_cov = 1/logdet_inv_cov
    return K_cov_inv, logdet_cov




def cov_power_plus_tsv(q_n, q_dist, alpha, gamma,  alpha2,  H_mat, tsv_mat, q0 = 10000.0):
    """Compute precision matrix for visibility considering powerlaw typye power spectrum

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        alpha (float): value determines the height of power spectrun at q0 (minimum visibility distance)
        gamma (float): power law index (q_k/q_0)**(-gamma)
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        K_cov_inv (ndarray): 2d array for precision matrix for visibility
        logdet_cov (float): determinant of covariance matrix
    """

    power_qk = alpha * (q_n/q0)**(-gamma)
    power_diag = np.diag(1/power_qk**2)
    K_tsv = alpha2 * tsv_mat
    K_cov_inv = H_mat.T@(power_diag+K_tsv)@H_mat
    (sign, logdet_inv_cov) = np.linalg.slogdet(K_cov_inv)
    logdet_cov = 1/logdet_inv_cov
    return K_cov_inv, logdet_cov

def cov_power_multifreq_gaussian_process(q_n, q_dist, theta, n_comp, H_mat):
    """ Calculate convariance matrix and its inverse

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): Number of components for Taylor expansions in frequency direction
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        inv_K_block_diag (ndarray): 2d precision matrix for model prior. This is used as prior information. 
        lot_det_cov_large (float): 2d precision matrix for model prior. This is used as prior information. 

    """

    H_large = []
    gamma_arr = theta[:n_comp]
    alpha_arr = theta[n_comp:]
    comp_now = len(alpha_arr)
    K_cov_inv_large = []
    lot_det_cov_large = 1

    for i in range(len(alpha_arr)):
        K_cov_inv, logdet_cov = cov_power(q_n, alpha_arr[i], gamma_arr[i], H_mat)
        lot_det_cov_large *= logdet_cov
        K_cov_inv_large.append(K_cov_inv)
    inv_K_block_diag = block_diag(*K_cov_inv_large)

    return inv_K_block_diag, lot_det_cov_large

def cov_power_multifreq(q_n, theta, n_comp, H_mat):
    """ Calculate convariance matrix and its inverse

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): Number of components for Taylor expansions in frequency direction
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        inv_K_block_diag (ndarray): 2d precision matrix for model prior. This is used as prior information. 
        lot_det_cov_large (float): 2d precision matrix for model prior. This is used as prior information. 

    """

    H_large = []
    gamma_arr = theta[:n_comp]
    alpha_arr = theta[n_comp:]
    comp_now = len(alpha_arr)
    K_cov_inv_large = []
    lot_det_cov_large = 1

    for i in range(len(alpha_arr)):
        K_cov_inv, logdet_cov = cov_power(q_n, alpha_arr[i], gamma_arr[i], H_mat)
        lot_det_cov_large *= logdet_cov
        K_cov_inv_large.append(K_cov_inv)
    inv_K_block_diag = block_diag(*K_cov_inv_large)

    return inv_K_block_diag, lot_det_cov_large

def log_evidence_cov_power_alpha(theta, q_n, vis_obs, sigma_vis):
    gamma = theta[0]
    alpha = theta[1]
    power_qk = alpha * (q_n/np.min(q_n))**(-gamma)
    return - 0.5 * np.sum(((vis_obs - power_qk)/sigma_vis)**2)

def log_evidence_cov_power_alpha_gp(theta, q_n, vis_obs, sigma_vis, q0=10**4):
    gamma = theta[0]
    alpha = theta[1]
    gamma2 = theta[2]
    alpha2 = theta[3]
    power_qk = alpha * (q_n/q0)**(-gamma)
    #power_qk = gauss_model_for_log(np.log(q_n), alpha, gamma, np.log(q0)) 
    #K_vis = alpha2 * np.exp(-(q_dist/gamma2)**2/2)

    return - 0.5 * np.sum(((np.abs(vis_obs) - power_qk)/sigma_vis)**2)

def log_evidence_cov_power_alpha_tsv(theta, q_n, vis_obs, sigma_vis, q0=10**4):
    gamma = theta[0]
    alpha = theta[1]
    alpha2 = theta[2]
    power_qk = alpha * (q_n/q0)**(-gamma)
    #power_qk = gauss_model_for_log(np.log(q_n), alpha, gamma, np.log(q0)) 
    #K_vis = alpha2 * np.exp(-(q_dist/gamma2)**2/2)

    return - 0.5 * np.sum(((np.abs(vis_obs) - power_qk)/sigma_vis)**2)


def return_logevidence_multifreq_power(theta, N, q_n, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp, H_mat):
    """ Compute log evidence

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        N (int): number of radial points in model
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        log_evidence (float): log evidence 
    """

    K_cov_inv, logdet_cov = cov_power_multifreq(q_n,theta, n_comp, H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside)
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    log_det = logdet_cov + logdet_for_sigma_d +  logdet_mat_inside
    inside_exp =-0.5 * d_A_minus1_d + 0.5 * V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d     
    log_evidence = - 0.5 * N * np.log(2*np.pi) - 0.5 * log_det  +  inside_exp
    return log_evidence

def return_logevidence_multifreq_power_plus(theta, N, q_n, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp, H_mat, q_bin, vis_bin, sigma_bin):
    """ Compute log evidence

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        N (int): number of radial points in model
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        log_evidence (float): log evidence 
    """

    K_cov_inv, logdet_cov = cov_power_multifreq(q_n,theta, n_comp, H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside)
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    log_det = logdet_cov + logdet_for_sigma_d +  logdet_mat_inside
    inside_exp =-0.5 * d_A_minus1_d + 0.5 * V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
    evidence_powerlaw = log_evidence_cov_power_alpha(theta, q_bin, vis_bin, sigma_bin)    
    log_evidence = - 0.5 * N * np.log(2*np.pi) - 0.5 * log_det  +  inside_exp + evidence_powerlaw 
    return log_evidence

def return_logevidence_multifreq_power_plus_gp(theta, N, q_n, q_dist, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp, H_mat, q_bin, vis_bin, sigma_bin):
    """ Compute log evidence

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        N (int): number of radial points in model
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        log_evidence (float): log evidence 
    """

    K_cov_inv, logdet_cov = cov_power_plus_gp(q_n, q_dist, theta[1], theta[0], theta[3], theta[2], H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside)
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    log_det = logdet_cov + logdet_for_sigma_d +  logdet_mat_inside
    inside_exp =-0.5 * d_A_minus1_d + 0.5 * V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
    evidence_powerlaw = log_evidence_cov_power_alpha_gp(theta, q_bin, vis_bin, sigma_bin)    
    log_evidence = - 0.5 * N * np.log(2*np.pi) - 0.5 * log_det  +  inside_exp + evidence_powerlaw 
    return log_evidence

def return_logevidence_multifreq_power_plus_tsv(theta, N, q_n, q_dist, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp, H_mat, q_bin, vis_bin, sigma_bin, tsv_mat):
    """ Compute log evidence

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        N (int): number of radial points in model
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        log_evidence (float): log evidence 
    """

    K_cov_inv, logdet_cov = cov_power_plus_tsv(q_n, q_dist, theta[1], theta[0],  theta[2], H_mat, tsv_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside)
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    log_det = logdet_cov + logdet_for_sigma_d +  logdet_mat_inside
    inside_exp =-0.5 * d_A_minus1_d + 0.5 * V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
    evidence_powerlaw = log_evidence_cov_power_alpha_tsv(theta, q_bin, vis_bin, sigma_bin)  
    log_evidence = - 0.5 * N * np.log(2*np.pi) - 0.5 * log_det  +  inside_exp + evidence_powerlaw 
    return log_evidence




def log_prior_power_multifreq(theta, n_comp, vis_max_from_obs, linear_prior = True):
    """ Compute log prior

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): number of Taylor components 
        vis_max_from_obs (float): maximum value for visibility

    Returns:
        log_prior_sum (theta): log prior
    """

    gamma_arr = theta[:n_comp]
    alpha_arr = theta[n_comp:]
    arcsecond = 1/206265.0
    log_prior_sum = 0

    for alpha_now in alpha_arr:

        if 0.1 * alpha_now  < alpha_now < 5 * vis_max_from_obs :
            continue
        else:
            return -np.inf    

    for gamma_now in gamma_arr:

        if 0.7 < gamma_now :
            continue
        else:
            return -np.inf

    return log_prior_sum

def log_prior_power_multifreq_gp(theta, n_comp, vis_max_from_obs, linear_prior = True):
    """ Compute log prior

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): number of Taylor components 
        vis_max_from_obs (float): maximum value for visibility

    Returns:
        log_prior_sum (theta): log prior
    """

    gamma = theta[0]
    alpha = theta[1]
    gamma2 = theta[2]
    alpha2 = theta[3]

    alpha_arr = theta[n_comp:]
    arcsecond = 1/206265.0
    log_prior_sum = 0


    if 0.5 * vis_max_from_obs  < alpha < 5 * vis_max_from_obs :
        log_prior_sum += 0
    else:
        return -np.inf    

    """
    if 0.7 < gamma :
        log_prior_sum += 0
    else:
        return -np.inf

    """
    if 10 < gamma<10**3 :
        log_prior_sum += np.log(1.0/gamma) 
    else:
        return -np.inf


    if 10**-12 < alpha2 <10:
        log_prior_sum += np.log(1.0/alpha2) 
    else:
        return -np.inf

    if 100 < gamma2 < 10**5:
        log_prior_sum += 0

    else:
        return -np.inf


    return log_prior_sum


def log_prior_power_multifreq_tsv(theta, n_comp, vis_max_from_obs, linear_prior = True):
    """ Compute log prior

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): number of Taylor components 
        vis_max_from_obs (float): maximum value for visibility

    Returns:
        log_prior_sum (theta): log prior
    """

    gamma = theta[0]
    alpha = theta[1]
    alpha2 = theta[2]

    alpha_arr = theta[n_comp:]
    arcsecond = 1/206265.0
    log_prior_sum = 0


    if 0.5 * vis_max_from_obs  < alpha < 5 * vis_max_from_obs :
        log_prior_sum += 0
    else:
        return -np.inf    

    """
    if 0.7 < gamma :
        log_prior_sum += 0
    else:
        return -np.inf

    """
    if 0.3 < gamma<1.5:
        log_prior_sum += np.log(1.0/gamma) 
    else:
        return -np.inf


    if 10**(-10) < alpha2:
        log_prior_sum += np.log(1.0/alpha2) 
    else:
        return -np.inf


    return log_prior_sum


def log_probability_power_multifreq(theta, N_d, q_n, H_mat_for_model, vis_max,  V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d,  n_comp, q_bin, vis_bin, sigma_bin):
    """ Compute log posterior 

    Args:
        theta (ndarray): 1d array for parameters. theta[0] is gamma, and theta[1:] are alpha parameters
        N_d (int): number of radial points in model
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
    Returns:
        log_pos (float): log posterior
    """

    lp = log_prior_power_multifreq(theta, n_comp, vis_max)
    if not np.isfinite(lp):
        return -np.inf
    log_pos = lp +return_logevidence_multifreq_power_plus(theta, N_d, q_n, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp, H_mat_for_model, q_bin, vis_bin, sigma_bin)

    return log_pos

def log_probability_power_multifreq_gp(theta, N_d, q_n, q_dist, H_mat_for_model, vis_max,  V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d,  n_comp, q_bin, vis_bin, sigma_bin):
    """ Compute log posterior 

    Args:
        theta (ndarray): 1d array for parameters. theta[0] is gamma, and theta[1:] are alpha parameters
        N_d (int): number of radial points in model
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
    Returns:
        log_pos (float): log posterior
    """

    lp = log_prior_power_multifreq_gp(theta, n_comp, vis_max)
    if not np.isfinite(lp):
        return -np.inf
    log_pos = lp +return_logevidence_multifreq_power_plus_gp(theta, N_d, q_n, q_dist, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp, H_mat_for_model, q_bin, vis_bin, sigma_bin)

    return log_pos

def log_probability_power_multifreq_tsv(theta, N_d, q_n, q_dist, H_mat_for_model, vis_max,  V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d,  n_comp, q_bin, vis_bin, sigma_bin, tsv_mat):
    """ Compute log posterior 

    Args:
        theta (ndarray): 1d array for parameters. theta[0] is gamma, and theta[1:] are alpha parameters
        N_d (int): number of radial points in model
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
    Returns:
        log_pos (float): log posterior
    """

    lp = log_prior_power_multifreq_tsv(theta, n_comp, vis_max)
    if not np.isfinite(lp):
        return -np.inf
    log_pos = lp +return_logevidence_multifreq_power_plus_tsv(theta, N_d, q_n, q_dist, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp, H_mat_for_model, q_bin, vis_bin, sigma_bin, tsv_mat)

    return log_pos

def sample_radial_profile_power_multifreq_gp(q_n, q_dist, H_mat_for_model, theta, n_comp, nu_arr, nu0, V_A_minus1_U, V_A_minus1_d):
    """ Take one sample for model given hyper parameters

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): Number of components for Taylor expansions in frequency direction
        nu_arr (ndarray): 1d array contaitning frequency. Number of frequencies is equal to number of 
            elements for first axis of "q_dist_freq"
        nu0 (float): reference frequency. Frequently used as form of ((nu-nu0)/nu0)**alpha
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
    Returns:
        sample_one (ndarray): 1d array for one sample of model.
            Taylors coefficients are stored in order (0-th, 1-st, ..)
        mean_gaussian (ndarray): 1d array for mean of model.
            Taylors coefficient vectors are stored in order (0-th, 1-st, ..)
        I_mfreq (ndarray): 2d array for radial profile for nu_arr
    """

    K_cov_inv, logdet = cov_power_plus_gp(q_n, q_dist, theta[1], theta[0], theta[3], theta[2], H_mat_for_model)
    mat_inside = K_cov_inv + V_A_minus1_U    
    mat_inside_inv = np.linalg.inv(mat_inside)
    mean_gaussian =  np.dot(mat_inside_inv, V_A_minus1_d)
    sample_one = np.random.multivariate_normal(mean_gaussian, mat_inside_inv)

    I_mfreq = []
    n_element = int(len(sample_one)/n_comp)
    for i in range(len(nu_arr)):
        I_mfreq_nu = np.zeros(n_element)
        for j in range(n_comp):
            I_mfreq_nu += (  ( ( nu_arr[i] - nu0)/nu0)**j) *sample_one[j*n_element:(j+1)*n_element]
        I_mfreq.append(I_mfreq_nu)
    I_mfreq = np.array(I_mfreq)
    return sample_one, mean_gaussian, I_mfreq

def sample_radial_profile_power_multifreq_tsv(q_n, q_dist, H_mat_for_model, theta, n_comp, nu_arr, nu0, V_A_minus1_U, V_A_minus1_d, tsv_mat):
    """ Take one sample for model given hyper parameters

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): Number of components for Taylor expansions in frequency direction
        nu_arr (ndarray): 1d array contaitning frequency. Number of frequencies is equal to number of 
            elements for first axis of "q_dist_freq"
        nu0 (float): reference frequency. Frequently used as form of ((nu-nu0)/nu0)**alpha
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
    Returns:
        sample_one (ndarray): 1d array for one sample of model.
            Taylors coefficients are stored in order (0-th, 1-st, ..)
        mean_gaussian (ndarray): 1d array for mean of model.
            Taylors coefficient vectors are stored in order (0-th, 1-st, ..)
        I_mfreq (ndarray): 2d array for radial profile for nu_arr
    """
    K_cov_inv, logdet = cov_power_plus_tsv(q_n, q_dist, theta[1], theta[0], theta[2], H_mat_for_model, tsv_mat)
    mat_inside = K_cov_inv + V_A_minus1_U    
    mat_inside_inv = np.linalg.inv(mat_inside)
    mean_gaussian =  np.dot(mat_inside_inv, V_A_minus1_d)
    sample_one = np.random.multivariate_normal(mean_gaussian, mat_inside_inv)

    I_mfreq = []
    n_element = int(len(sample_one)/n_comp)
    for i in range(len(nu_arr)):
        I_mfreq_nu = np.zeros(n_element)
        for j in range(n_comp):
            I_mfreq_nu += (  ( ( nu_arr[i] - nu0)/nu0)**j) *sample_one[j*n_element:(j+1)*n_element]
        I_mfreq.append(I_mfreq_nu)
    I_mfreq = np.array(I_mfreq)
    return sample_one, mean_gaussian, I_mfreq


def initial_for_power_multifreq(alpha_value, n_alpha, nwalker):
    """ Generate random walkers for emcee

    Args:
        alpha_value (float): value for 
        n_alpha (int): number of alpha (Taylor coefficients)
        nwalker (int): number of walkers
    Returns:
        para_arr_arr (ndarray): 2d array (nwalkers, 1+n_alpha) containing parameters values
    """ 

    para_arr_arr = []
    for i_gamma in range(n_alpha):
        gamma_arr = 0.8 + 0.01 * np.random.rand(nwalker)
        para_arr_arr.append(gamma_arr)
    for i in range(n_alpha):
        alpha_arr =alpha_value * (1 +  0.01* np.random.rand(nwalker))
        para_arr_arr.append(alpha_arr)
    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr

def initial_for_power_multifreq_gp(alpha_value, n_alpha, nwalker):
    """ Generate random walkers for emcee

    Args:
        alpha_value (float): value for 
        n_alpha (int): number of alpha (Taylor coefficients)
        nwalker (int): number of walkers
    Returns:
        para_arr_arr (ndarray): 2d array (nwalkers, 1+n_alpha) containing parameters values
    """ 

    para_arr_arr = []
    for i_gamma in range(n_alpha):
        gamma_arr = 1 + (10**3 - 10) * np.random.rand(nwalker)
        para_arr_arr.append(gamma_arr)

    """
    for i_gamma in range(n_alpha):
        gamma_arr = 0.8 + 0.01 * np.random.rand(nwalker)
        para_arr_arr.append(gamma_arr)        
    """
    for i in range(n_alpha):
        alpha_arr =alpha_value * (1 +  0.01* np.random.rand(nwalker))
        para_arr_arr.append(alpha_arr)

    for i_gamma in range(n_alpha):
        gamma_arr = 100 + (10**5 - 100) * np.random.rand(nwalker)
        para_arr_arr.append(gamma_arr)

    for i in range(n_alpha):
        alpha_arr = 10**(-12 +13 * np.random.rand(nwalker))
        para_arr_arr.append(alpha_arr)


    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr

def initial_for_power_multifreq_tsv(alpha_value, n_alpha, nwalker):
    """ Generate random walkers for emcee

    Args:
        alpha_value (float): value for 
        n_alpha (int): number of alpha (Taylor coefficients)
        nwalker (int): number of walkers
    Returns:
        para_arr_arr (ndarray): 2d array (nwalkers, 1+n_alpha) containing parameters values
    """ 

    para_arr_arr = []
    for i_gamma in range(n_alpha):
        gamma_arr = 0.8 + 0.1 * np.random.rand(nwalker)
        para_arr_arr.append(gamma_arr)

    """
    for i_gamma in range(n_alpha):
        gamma_arr = 0.8 + 0.01 * np.random.rand(nwalker)
        para_arr_arr.append(gamma_arr)        
    """
    for i in range(n_alpha):
        alpha_arr =alpha_value * (1 +  0.01* np.random.rand(nwalker))
        para_arr_arr.append(alpha_arr)


    for i in range(n_alpha):
        alpha_arr = 10**(10+5 * np.random.rand(nwalker))
        para_arr_arr.append(alpha_arr)


    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr



def log_prior_multi_freq_geo(theta, log10_alpha_mean, n_comp, alpha_log_width = 8, linear_prior = True):
    """ Compute log prior

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        log10_alpha_mean: (float): log value of mean alpha for prior
        n_comp (int): number of Taylor components 
        alpha_log_width (float): width of log prior

    Returns:
        log_prior_sum (theta): log prior
    """


    gamma_arr = [theta[0]]
    alpha_arr = [theta[1]]
    inc = theta[2]
    pa = theta[3]
    arcsecond = 1/206265.0
    log_prior_sum = 0
    for gamma_now in gamma_arr:

        if MIN_RSCALE < gamma_now < MAX_RSCALE :
            if linear_prior:
                log_prior_sum+= np.log(1.0/(0.5 * arcsecond- 0.001* arcsecond))
            else:
                log_prior_sum+= np.log(1.0/gamma_now)

        else:
            return -np.inf

    for alpha_now in alpha_arr:
        if 10**(log10_alpha_mean-alpha_log_width) < alpha_now < 10**(log10_alpha_mean+alpha_log_width) :
            log_prior_sum += np.log(1.0/alpha_now) 
        else:
            return -np.inf
    if -0.5 * np.pi < inc < 0.5 * np.pi:
        log_prior_sum += np.log(np.abs(np.sin(inc)))
    else:
        return -np.inf

    if 0 < pa < np.pi:
        log_prior_sum += 0
    else:
        return -np.inf

    return log_prior_sum


def return_logevidence_multifreq_geo(theta, N, r_dist, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp):
    """ Compute log evidence

    Args:
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        N (int): number of radial points in model
        r_dist (ndarray): 2d array containing distnace between different pixels 
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        n_comp (int): number of Taylor components 
    Returns:
        log_evidence (float): log evidence 
    """

    K_cov, K_cov_inv = cov_matern_multifreq(r_dist,theta, n_comp)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside)
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    (sign, logdet_cov)  = np.linalg.slogdet(K_cov)
    log_det = logdet_cov + logdet_for_sigma_d +  logdet_mat_inside
    inside_exp =-0.5 * d_A_minus1_d + 0.5 * V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d     
    log_evidence = - 0.5 * N * np.log(2*np.pi) - 0.5 * log_det  +  inside_exp

    return log_evidence
    
def log_probability_multifreq_geo(theta, N_d, r_dist, u_d, v_d, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, log10_alpha_mean, n_comp):
    """ Compute log posterior 

    Args:
        theta (ndarray): 1d array for parameters. theta[0] is gamma, and theta[1:] are alpha parameters
        N (int): number of radial points in model
        r_dist (ndarray): 2d array containing distnace between different pixels 
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
        log10_alpha_mean: (float): log value of mean alpha for prior
        n_comp (int): number of Taylor components 
    Returns:
        log_pos (float): log posterior
    """

    lp = log_prior_multifreq(theta, log10_alpha_mean, n_comp)
    H_
    if not np.isfinite(lp):
        return -np.inf
    log_pos = lp +return_logevidence_multifreq(theta, N_d, r_dist, V_A_minus1_U, V_A_minus1_d, d_A_minus1_d, logdet_for_sigma_d, n_comp)

    return log_pos

def make_initial_random_multifreq_geo(log10_alpha_min, log10_alpha_max, gamma_min, gamma_max, n_alpha, nwalker):
    """ Generate random walkers for emcee

    Args:
        log10_alpha_min (float): min value for log alpha. "log10_alpha_min" < log alpha < log10_alpha_max
        log10_alpha_max (float): max value for log alpha. log10_alpha_min < log alpha < "log10_alpha_max"
        gamma_min (float): min value for gamma
        gamma_max (float): max value for gamma
        n_alpha (int): number of alpha (Taylor coefficients)
        nwalker (int): number of walkers
    Returns:
        para_arr_arr (ndarray): 2d array (nwalkers, 1+n_alpha) containing parameters values
    """ 

    para_arr_arr = []
    for i_gamma in range(n_alpha):
        gamma_arr = gamma_min + (gamma_max - gamma_min) * np.random.rand(nwalker)
        para_arr_arr.append(gamma_arr)
    for i in range(n_alpha):
        alpha_arr = 10**(log10_alpha_min + (log10_alpha_max - log10_alpha_min) * np.random.rand(nwalker))
        para_arr_arr.append(alpha_arr)
    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr


def sample_radial_profile_power_multifreq_geo(q_n, H_mat_for_model, theta, n_comp, nu_arr, nu0, V_A_minus1_U, V_A_minus1_d):
    """ Take one sample for model given hyper parameters

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        theta (ndarray): 1d array for parameters. theta[:n_comp] is gamma array, and theta[n_comp:] are alpha parameters
        n_comp (int): Number of components for Taylor expansions in frequency direction
        nu_arr (ndarray): 1d array contaitning frequency. Number of frequencies is equal to number of 
            elements for first axis of "q_dist_freq"
        nu0 (float): reference frequency. Frequently used as form of ((nu-nu0)/nu0)**alpha
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
    Returns:
        sample_one (ndarray): 1d array for one sample of model.
            Taylors coefficients are stored in order (0-th, 1-st, ..)
        mean_gaussian (ndarray): 1d array for mean of model.
            Taylors coefficient vectors are stored in order (0-th, 1-st, ..)
        I_mfreq (ndarray): 2d array for radial profile for nu_arr
    """

    K_cov_inv, logdet = cov_power_multifreq(q_n,theta, n_comp, H_mat_for_model)
    mat_inside = K_cov_inv + V_A_minus1_U    
    mat_inside_inv = np.linalg.inv(mat_inside)
    mean_gaussian =  np.dot(mat_inside_inv, V_A_minus1_d)
    sample_one = np.random.multivariate_normal(mean_gaussian, mat_inside_inv)

    I_mfreq = []
    n_element = int(len(sample_one)/n_comp)
    for i in range(len(nu_arr)):
        I_mfreq_nu = np.zeros(n_element)
        for j in range(n_comp):
            I_mfreq_nu += (  ( ( nu_arr[i] - nu0)/nu0)**j) *sample_one[j*n_element:(j+1)*n_element]
        I_mfreq.append(I_mfreq_nu)
    I_mfreq = np.array(I_mfreq)
    return sample_one, mean_gaussian, I_mfreq


