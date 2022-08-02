import os 
#os.environ["OMP_NUM_THREADS"] = "1"

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
from multiprocessing import Pool, freeze_support


ARCSEC_TO_RAD= 1/206265.0


"""
N_RAD = 30
D_PIX = 0.10 * ARCSEC_TO_RAD
MAXRAD = N_RAD* D_PIX 
DO_MCMC = True
DO_TEST_CALC = False
DO_SAMPLING = True
MCMC_RUN = 10000
SAMPLE_NUM = 50
NWALKER = 32
MIN_RSCALE =  0.1 * ARCSEC_TO_RAD 
MAX_RSCALE = 4.0 * ARCSEC_TO_RAD 
D_OFFSET = 0.9 * ARCSEC_TO_RAD
LOGALPHAMIN=-13
LOGALPHAMAX=-4
"""

def cov_power(q_n, a_power, alpha, gamma, H_mat, q0=10**4):
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
    power_qk = a_power * (q_n/q0)**(-gamma)
    power_diag = (1/alpha) * np.diag(1/power_qk**2)
    K_cov_inv = H_mat.T@power_diag@H_mat
    (sign, logdet_inv_cov) = np.linalg.slogdet(K_cov_inv)
    logdet_cov = 1/logdet_inv_cov
    return K_cov_inv, logdet_cov

def matern(r_dist, alpha, gamma):
    ins = (np.sqrt(3)* r_dist/gamma)
    K= alpha *(1 + ins ) * np.exp(-ins)
    return K

def cov_matern(r_dist, theta):
    """ Calculate convariance matrix and its inverse

    Args:
        r_dist (ndarray): 2d array containing distnace between different pixels 
        theta (ndarray): 1d array for parameters.
    Returns:
        K_cov(ndarray): 2d convariance matrix for model prior. This is used as prior information. 
        K_cov_inv(ndarray): 2d precision matrix for model prior. This is used as prior information. 

    """
    gamma = theta[0]
    alpha = theta[1]
    K_cov = matern(r_dist, alpha, gamma)
    K_cov_inv= np.linalg.inv(K_cov)
    return K_cov, K_cov_inv


def RBF_add_noise(obst, alpha, gamma, simga = 1e-7):
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    nx, ny = np.shape(obst)
    K=alpha * (np.exp(-(Dt/gamma)**2/2) + simga * np.identity(nx))
    K_inv= np.linalg.inv(K)
    #(sign, logdet_inv_cov) = np.linalg.slogdet(K_inv)
    return K, K_inv#, log_det_K



def make_2d_mat_for_dist(r_arr):
    r_pos_tile = np.tile(r_arr,(len(r_arr),1)) 
    r_dist = ((r_pos_tile - r_pos_tile.T)**2)**0.5
    return r_dist
def load_obsdata(file_name):
    """ Loading function for multi-freq data. This function deprojects visibility using 
    assumed inclination and positiona angle. 
    
    Args:
        file_name (str): file name for data
        inc (float): inclination of object
        pa (flaot): position angle of object
    
    Returns:
        q_dist (ndarray): 1d array containing visibility distance in radial direction
        vis_d_real (ndarray): 1d array contatining real part of visibility in radial direction
        freq_d (ndarray): 1d array containig frequency 
    """


    data = np.load(file_name)
    v_d = data["v_obs"]
    u_d = data["u_obs"]
    vis_d = data["vis_obs"]
    wgt_d = data["wgt_obs"]
    return u_d, v_d, vis_d, wgt_d

def diag_multi(diag_sigma, mat):
    return (diag_sigma * mat.T).T

def make_hankel_matrix(q, R_out, N, dpix):
    j_nplus = jn_zeros(0, N+1)
    j_nk = jn_zeros(0, N + 1)
    j_nk, j_nN = j_nk[:-1], j_nk[-1]
    r_pos = R_out * j_nk/j_nN
    factor = (1/dpix**2) * 4 * np.pi * R_out**2 / (j_nN**2)
    scale_factor = 1/(j1(j_nk) ** 2)
    H_mat = factor * scale_factor * j0(np.outer(q, 2 * np.pi * r_pos))
    return H_mat

def obs_model_comparison(I_model, u_d, v_d, cosi, pa, delta_x, delta_y, I_0_input, d_data, R_out, N, dpix):
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d_before = -cos_pa * u_d + sin_pa *v_d
    v_new_d = -sin_pa * u_d - cos_pa *v_d
    u_new_d = u_new_d_before * cosi
    q_dist = (u_new_d**2 + v_new_d **2)**0.5    
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_cos_inv = np.cos(2 * np.pi * (delta_x * u_d + delta_y * v_d))
    diag_mat_sin_inv = np.sin(2 * np.pi * (delta_x * u_d + delta_y * v_d))

    n_d = int(len(d_data)/2)
    d_real = d_data[:n_d]
    d_imag = d_data[n_d:]
    d_real_mod = d_real * diag_mat_cos_inv  - d_imag * diag_mat_sin_inv
    d_imag_mod = + d_real * diag_mat_sin_inv + d_imag * diag_mat_cos_inv

    d_real_for_point_source = I_0_input * np.ones(int(len(d_data)/2))
    H_mat = make_hankel_matrix(q_dist, R_out, N, dpix)
    vis_model = np.dot(H_mat, I_model) + d_real_for_point_source
    vis_model_imag = np.zeros(np.shape(vis_model))

    return H_mat, q_dist, d_real_mod, d_imag_mod, vis_model, vis_model_imag, u_new_d_before, v_new_d

def make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, dpix):

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d = -cos_pa * u_d + sin_pa *v_d
    v_new_d = -sin_pa * u_d - cos_pa *v_d
    u_new_d = u_new_d * cosi
    q_dist = (u_new_d**2 + v_new_d **2)**0.5
    H_mat = make_hankel_matrix(q_dist, R_out, N, dpix)
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat = np.append(diag_mat_cos, diag_mat_sin)
    H_mat_off1 = diag_multi(diag_mat_cos, H_mat)
    H_mat_off2 = diag_multi(diag_mat_sin, H_mat)
    H_mat_all = np.concatenate([H_mat_off1, H_mat_off2])
    return H_mat_all

def prepare_calculation_for_mcmc(r_pos, d, sigma_mat_diag, file_name =None, rewrite =True):
    N_d = len(d)
    r_dist = make_2d_mat_for_dist(r_pos)
    d_A_minus1_d = np.sum(sigma_mat_diag* d * d)
    logdet_for_sigma_d = np.sum(np.log(sigma_mat_diag))
    return N_d, r_dist, d_A_minus1_d, logdet_for_sigma_d 

def chi_for_positional_off(theta, d_data, sigma_d, u_d, v_d, return_im = False):
    delta_x = theta[0] 
    delta_y = theta[1]
    diag_mat_cos_inv = np.cos(2 * np.pi * (delta_x * u_d + delta_y * v_d))
    diag_mat_sin_inv = np.sin(2 * np.pi * (delta_x * u_d + delta_y * v_d))

    n_d = int(len(d_data)/2)
    d_real = d_data[:n_d]
    d_imag = d_data[n_d:]
    simga_real = sigma_d[:n_d]
    sigma_imag = sigma_d[n_d:]
    d_imag_mod = + d_real * diag_mat_sin_inv + d_imag * diag_mat_cos_inv
    d_real_mod = d_real * diag_mat_cos_inv  - d_imag * diag_mat_sin_inv


    if return_im:
        return d_real_mod, d_imag_mod
    else:
        return np.sum((d_imag_mod**2) * sigma_imag )




def determine_positional_offsets(d_data, sigma_d, u_d, v_d, delta_pos = 1, n_try = 100):
    #res = minimize(chi_for_positional_off, [-0.354 *ARCSEC_TO_RAD, -0.164 * ARCSEC_TO_RAD], method='BFGS',\
    #    args=(d_data, u_d, v_d), bounds =( (-ARCSEC_TO_RAD, ARCSEC_TO_RAD), (-ARCSEC_TO_RAD, ARCSEC_TO_RAD)))
    best_positions = []
    chi_imag = []
    delta_x_arr =  2 * (np.random.rand(n_try) - 0.5) * delta_pos * ARCSEC_TO_RAD
    delta_y_arr = 2 * (np.random.rand(n_try) - 0.5) *  delta_pos * ARCSEC_TO_RAD


    for n_i in range(n_try):
        res = minimize(chi_for_positional_off, [ delta_x_arr[n_i],delta_y_arr[n_i]], method='L-BFGS-B',\
            args=(d_data, sigma_d, u_d, v_d), bounds =( (-ARCSEC_TO_RAD, ARCSEC_TO_RAD), (-ARCSEC_TO_RAD, ARCSEC_TO_RAD)))
        d_real_mod, d_imag_mod  = chi_for_positional_off(res.x, d_data, sigma_d, u_d, v_d, True)
        best_positions.append(res.x)
        chi_imag.append(np.sum(d_imag_mod**2))
        #print(res.x/ARCSEC_TO_RAD)
    best_positions = np.array(best_positions)
    chi_imag = np.array(chi_imag)
    arg_best = np.argmin(chi_imag)
    #print (best_positions[arg_best]/ARCSEC_TO_RAD)
    return best_positions[arg_best]

def delta_x_for_d_subtraction(d, u_d, v_d, theta): 
    delta_x = theta[4]
    delta_y = theta[5]
    I_0 = theta[6]
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_all = I_0 * np.append(diag_mat_cos, diag_mat_sin)
    return d-diag_mat_all



def log_prior_geo(theta, log10_alpha_mean, min_scale, max_scale, alpha_log_width = 8, delta_pos = 1.0, min_index = 1, max_index = 7, linear_prior = True, cov = "RBF"):
    """ Compute log prior

    Args:
        theta (ndarray): 1d array for parameters. 
        log10_alpha_mean: (float): log value of mean alpha for prior
        alpha_log_width (float): width of log prior

    Returns:
        log_prior_sum (theta): log prior
    """
    log_prior_sum = 0
    if cov=="RBF":

        gamma_arr = [theta[0]]
        alpha_arr = [theta[1]]
        cosi = theta[2]
        pa = theta[3]
        delta_x = theta[4]
        delta_y = theta[5]
        I_0 = theta[6]
        arcsecond = 1/206265.0
        for gamma_now in gamma_arr:

            if min_scale <= gamma_now <= max_scale:
                log_prior_sum += 0

            else:
                return -np.inf

        for alpha_now in alpha_arr:
            if  10**(log10_alpha_mean-alpha_log_width) <= alpha_now <= 10**(log10_alpha_mean+alpha_log_width) :
                log_prior_sum += np.log(1.0/alpha_now) 
            else:
                return -np.inf
        if 0 <= cosi <= 1:
            log_prior_sum += 0
        else:
            return -np.inf

        if 0 <= pa <= np.pi:
            log_prior_sum += 0
        else:
            return -np.inf

        if -delta_pos* arcsecond <= delta_x <= delta_pos * arcsecond:
            log_prior_sum += 0
        else:
            return -np.inf

        if -delta_pos * arcsecond <= delta_y <= delta_pos * arcsecond:
            log_prior_sum += 0
        else:
            return -np.inf

        if 0<=I_0<1:
            log_prior_sum += 0
        else:
            return -np.inf

        return log_prior_sum

    if cov=="power":

        gamma_now = theta[0]
        a_power_now = theta[1]
        alpha_now = theta[2]
        cosi = theta[3]
        pa = theta[4]
        delta_x = theta[5]
        delta_y = theta[6]
        arcsecond = 1/206265.0

        if min_index< gamma_now < max_index:
            log_prior_sum += 0
        else:
            return -np.inf




        for alpha_now in alpha_arr:



            if 10**(log10_alpha_mean-alpha_log_width) < alpha_now < 10**(log10_alpha_mean+alpha_log_width) :
                log_prior_sum += np.log(1.0/alpha_now) 
            else:
                return -np.inf
        if 0 <cosi <1:
            log_prior_sum += 0
        else:
            return -np.inf

        if 0 < pa < np.pi:
            log_prior_sum += 0
        else:
            return -np.inf

        if -delta_pos* arcsecond < delta_x < delta_pos * arcsecond:
            log_prior_sum += 0
        else:
            return -np.inf

        if -delta_pos * arcsecond < delta_y < delta_pos * arcsecond:
            log_prior_sum += 0
        else:
            return -np.inf

        return log_prior_sum


def return_logevidence_geo(theta, r_dist,  H_mat, d,  sigma_d, cov = "RBF"):
    """ Compute log evidence

    Args:
        theta (ndarray): 1d array for parameters.
        N (int): number of radial points in model
        r_dist (ndarray): 2d array containing distnace between different pixels 
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
        d_A_minus1_d (float): value for d@A^-1@d
        logdet_for_sigma_d (float): log determinant of error matrix
    Returns:
        log_evidence (float): log evidence 
    """
    V_A_minus1_U = H_mat.T@diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*d)
    if cov=="RBF":
        K_cov, K_cov_inv = RBF_add_noise(r_dist,theta[1], theta[0])
    if cov=="power":
        K_cov, K_cov_inv = cov_power(q_dist,theta[2], theta[0], theta[1], H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside)
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    (sign, logdet_cov)  = np.linalg.slogdet(K_cov)
    log_det = logdet_cov + logdet_mat_inside
    log_evidence = - 0.5 * log_det  +  0.5 * V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
    return log_evidence

def utility_for_emcee(u_d, v_d, delta_x, delta_y, vis_d):
    n_d = int(len(vis_d)/2)

    diag_mat_cos_inv = np.cos( 2 * np.pi * (delta_x * u_d + delta_y * v_d))
    diag_mat_sin_inv = np.sin( 2 * np.pi * (delta_x * u_d + delta_y * v_d))

    d_real = vis_d[:n_d]
    d_imag = vis_d[n_d:]
    d_real_mod = d_real * diag_mat_cos_inv  - d_imag * diag_mat_sin_inv
    d_imag_mod = + d_real * diag_mat_sin_inv + d_imag * diag_mat_cos_inv

    return d_real_mod, d_imag_mod

def give_q_max(u_d, v_d, cosi, pa):

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d = -cos_pa * u_d + sin_pa *v_d
    v_new_d = -sin_pa * u_d - cos_pa *v_d
    u_new_d = cosi * u_new_d
    q_max = np.max(( (u_new_d)**2 + v_new_d**2)**0.5)
    return q_max


def give_q(u_d, v_d, cosi, pa):

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d = -cos_pa * u_d + sin_pa *v_d
    v_new_d = -sin_pa * u_d - cos_pa *v_d
    u_new_d = cosi * u_new_d
    q_d = ( (u_new_d)**2 + v_new_d**2)**0.5
    return q_d

def log_probability_geo_for_emcee(theta, N_d, r_dist, u_d, v_d, vis_d, sigma_d,  log10_alpha_mean, log10_alpha_wd, min_scale, max_scale, R_out, N, dpix):
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
    Returns:
        log_pos (float): log posterior
    """
    lp = log_prior_geo(theta, log10_alpha_mean,min_scale, max_scale, log10_alpha_wd)
    if not np.isfinite(lp):
        #print(-np.inf, -1e6, -1e6, -1e6, -1e6,-1e6)
        return -np.inf, -1e100, -1e100

    q_max_for_dpix = give_q_max(u_d, v_d, theta[2], theta[3])
    if dpix is None:
        dpix= 0.5/q_max_for_dpix
    R_out = N * dpix    
    r_n, jn, qmax, q_n = make_collocation_points(R_out, N)
    r_dist = make_2d_mat_for_dist(r_n)
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    H_mat = make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, dpix)
    d_offset =delta_x_for_d_subtraction(vis_d, u_d, v_d, theta)
    lhood = return_logevidence_geo(theta, r_dist, H_mat, d_offset, sigma_d)
    log_pos = lp +lhood
    #print(log_pos, lp, lhood, d_num, chi_sq, imag_sum_sq)
    #print(log_pos, lhood, lp, chi_sq)
    return log_pos, lhood, lp
    sample_mean = sample_mean_radial_profile(r_dist, theta, u_d, v_d, R_out, N, dpix,  vis_d, sigma_d, cov="RBF")
    vis_model = np.dot(H_mat, sample_mean)
    #print(np.sum(((vis_model - vis_d)**2)*sigma_d))
    chi_sq = np.sum((( (vis_model - vis_d)**2 )*sigma_d))
    d_real_mod, d_imag_mod = utility_for_emcee(u_d, v_d, delta_x, delta_y, vis_d)
    imag_sum_sq = np.sum(d_imag_mod**2)
    d_num = len(vis_d)

    lhood = return_logevidence_geo(theta, r_dist, H_mat, vis_d, sigma_d)
    log_pos = lp +lhood
    #print(log_pos, lp, lhood, d_num, chi_sq, imag_sum_sq)
    return log_pos, lp, lhood, d_num, chi_sq, imag_sum_sq

def log_probability_geo_for_minimizer(theta, N_d, r_dist, u_d, v_d, vis_d, sigma_d,  log10_alpha_mean, log10_alpha_wd, min_scale, max_scale, R_out, N, dpix):
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
    Returns:
        log_pos (float): log posterior
    """

    q_max_for_dpix = give_q_max(u_d, v_d, theta[2], theta[3])
    if dpix is None:
        dpix= 0.5/q_max_for_dpix    
    R_out = N * dpix    
    r_n, jn, qmax, q_n = make_collocation_points(R_out, N)
    r_dist = make_2d_mat_for_dist(r_n)
    lp = log_prior_geo(theta, log10_alpha_mean, min_scale, max_scale, log10_alpha_wd)
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    H_mat = make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, dpix)
    if not np.isfinite(lp):
        return -np.inf
    lhood = return_logevidence_geo(theta, r_dist, H_mat, vis_d, sigma_d)
    log_pos = lp +lhood
    #print(lp, lhood)
    return log_pos

def make_initial_geo_from_optimize(theta_init, log10_alpha_min, log10_alpha_max, gamma_min, gamma_max, delta_pos ,  nwalker):
    """ Generate random walkers for emcee

    Args:
        log10_alpha_min (float): min value for log alpha. "log10_alpha_min" < log alpha < log10_alpha_max
        log10_alpha_max (float): max value for log alpha. log10_alpha_min < log alpha < "log10_alpha_max"
        gamma_min (float): min value for gamma
        gamma_max (float): max value for gamma
        nwalker (int): number of walkers
    Returns:
        para_arr_arr (ndarray): 2d array (nwalkers, 1) containing parameters values
    """ 


    gamma_arr = theta_init[0] + ((gamma_max-gamma_min)/100) *  np.random.rand(nwalker)
    alpha_arr = 10**( np.log10(theta_init[1]) +0.1  * np.random.rand(nwalker))
    cosi_arr = theta_init[2] + 0.01 * np.random.rand(nwalker)  
    pa_arr = theta_init[3] + 0.01 * np.random.rand(nwalker) * np.pi  
    delta_x_arr = theta_init[4] + 0.01 *  np.random.rand(nwalker) * delta_pos
    delta_y_arr = theta_init[5] + 0.01 *  np.random.rand(nwalker) * delta_pos
    para_arr_arr = []
    para_arr_arr.append(gamma_arr) 
    para_arr_arr.append(alpha_arr)
    para_arr_arr.append(cosi_arr)
    para_arr_arr.append(pa_arr)
    para_arr_arr.append(delta_x_arr)
    para_arr_arr.append(delta_y_arr)
    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr

def make_initial_geo_offset(log10_alpha_min, log10_alpha_max, gamma_min, gamma_max, I_0_input, delta_pos ,nwalker,  x_est = None, y_est = None):
    """ Generate random walkers for emcee

    Args:
        log10_alpha_min (float): min value for log alpha. "log10_alpha_min" < log alpha < log10_alpha_max
        log10_alpha_max (float): max value for log alpha. log10_alpha_min < log alpha < "log10_alpha_max"
        gamma_min (float): min value for gamma
        gamma_max (float): max value for gamma
        nwalker (int): number of walkers
    Returns:
        para_arr_arr (ndarray): 2d array (nwalkers, 1) containing parameters values
    """ 


    gamma_arr = gamma_min + (gamma_max - gamma_min) * np.random.rand(nwalker)
    alpha_arr = 10**(log10_alpha_min + (log10_alpha_max - log10_alpha_min) * np.random.rand(nwalker))
    pa_arr = np.random.rand(nwalker) * np.pi  
    cosi_arr = np.random.rand(nwalker) 
    I0_arr = I_0_input + np.random.rand(nwalker) *0.0001
    if x_est is not None:
        delta_x_arr = x_est +  0.001  *  np.random.rand(nwalker) * delta_pos
        delta_y_arr = y_est  +  0.001  *  np.random.rand(nwalker) * delta_pos
    else:
        delta_x_arr = 0.0 * ARCSEC_TO_RAD  + 0.001 *  np.random.rand(nwalker) * delta_pos
        delta_y_arr = 0.0 * ARCSEC_TO_RAD  +  0.001  *  np.random.rand(nwalker) * delta_pos

    para_arr_arr = []
    para_arr_arr.append(gamma_arr) 
    para_arr_arr.append(alpha_arr)
    para_arr_arr.append(cosi_arr)
    para_arr_arr.append(pa_arr)
    para_arr_arr.append(delta_x_arr)
    para_arr_arr.append(delta_y_arr)
    para_arr_arr.append(I0_arr)
    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr

def sample_mean_radial_profile(r_dist, theta, u_d, v_d, R_out, N, dpix,  d, sigma_d, cov="RBF"):
    """ Take one sample for model given hyper parameters

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        theta (ndarray): 1d array for parameters.
        nu_arr (ndarray): 1d array contaitning frequency. Number of frequencies is equal to number of 
            elements for first axis of "q_dist_freq"
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
    Returns:
        sample_one (ndarray): 1d array for one sample of model.
            Taylors coefficients are stored in order (0-th, 1-st, ..)
        mean_gaussian (ndarray): 1d array for mean of model.
            Taylors coefficient vectors are stored in order (0-th, 1-st, ..)
        I_mfreq (ndarray): 2d array for radial profile for nu_arr
    """
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    H_mat = make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, dpix)
    V_A_minus1_U = H_mat.T@diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*d)
    if cov=="RBF":
        K, K_cov_inv = RBF_add_noise(r_dist,theta[1], theta[0])
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside)
    mean_gaussian =  np.dot(mat_inside_inv, V_A_minus1_d)
    sample_one = np.random.multivariate_normal(mean_gaussian, mat_inside_inv)
    return mean_gaussian


def sample_radial_profile(r_dist, theta, u_d, v_d, R_out, N, dpix,  d, sigma_d, cov="RBF"):
    """ Take one sample for model given hyper parameters

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        H_mat_for_model (ndarray): 2d array for design matrix converting intensity to visibility
        theta (ndarray): 1d array for parameters.
        nu_arr (ndarray): 1d array contaitning frequency. Number of frequencies is equal to number of 
            elements for first axis of "q_dist_freq"
        V_A_minus1_U (ndarray): 2d array for V@A^-1@U
        V_A_minus1_d (ndarray): 1d array for V@A^-1@d
    Returns:
        sample_one (ndarray): 1d array for one sample of model.
            Taylors coefficients are stored in order (0-th, 1-st, ..)
        mean_gaussian (ndarray): 1d array for mean of model.
            Taylors coefficient vectors are stored in order (0-th, 1-st, ..)
        I_mfreq (ndarray): 2d array for radial profile for nu_arr
    """
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    d_offset =delta_x_for_d_subtraction(d, u_d, v_d, theta)
    H_mat = make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, dpix)
    V_A_minus1_U = H_mat.T@diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*d_offset)
    if cov=="RBF":
        K, K_cov_inv = RBF_add_noise(r_dist,theta[1], theta[0])
    mat_inside = K_cov_inv + V_A_minus1_U    
    mat_inside_inv = np.linalg.inv(mat_inside)
    mean_gaussian =  np.dot(mat_inside_inv, V_A_minus1_d)
    sample_one = np.random.multivariate_normal(mean_gaussian, mat_inside_inv)
    return sample_one, H_mat

def make_collocation_points(R_out, N):
    """ Making collocation points for Hankel transformation

    Params:
        R_out (float): Upper limit on radial size of object 
        N (int): number of radial points in model
    Returns:
        r_n (ndarray): 1d radial positions of collocation points
        j_nN (ndarray): roots of the order 0 Bessel function of the first kind
        q_max (float): Maximum visibility distance
    """

    j_nk = jn_zeros(0, N + 1)
    j_nk, j_nN = j_nk[:-1], j_nk[-1]
    r_n = R_out * j_nk/j_nN
    q_n = j_nk/(2 * np.pi * R_out )
    q_max =j_nN /(2 * np.pi * R_out)
    return r_n, j_nN, q_max, q_n


def samples_for_plot(samples):

    samples_update = np.copy(samples)
    samples_update[:,0] = samples[:,0]/ARCSEC_TO_RAD
    samples_update[:,3] = samples[:,3] * 180/np.pi
    samples_update[:,1] = np.log10(samples[:,1])
    samples_update[:,2] = samples[:,2] 
    samples_update[:,4] = samples[:,4]/ARCSEC_TO_RAD
    samples_update[:,5] = samples[:,5]/ARCSEC_TO_RAD


    return samples_update

def sample_for_plot(sample):


    sample_update = np.copy(sample)
    sample_update[0] = sample[0]/ARCSEC_TO_RAD
    sample_update[3] = sample[3] * 180/np.pi
    sample_update[1] = np.log10(sample[1])
    sample_update[2] = sample[2] 
    sample_update[4] = sample[4]/ARCSEC_TO_RAD
    sample_update[5] = sample[5]/ARCSEC_TO_RAD
    return sample_update

def masking_data(H_mat, I_model, d_real, d_imag, sigma_real, sigma_imag, threshold = 5):

    model = np.dot(H_mat, I_model)
    len_d = int(len(model)/2)

    mask = (np.abs( (d_real - model)/sigma_real) > threshold ) + (np.abs( (d_imag)/sigma_imag) > threshold )

    return mask

def q_max_determine(u_d, v_d):
    cosi_arr = np.linspace(0,1,21)
    pa_arr = np.linspace(0,np.pi,30) 
    q_max_arr = []

    for cosi_now in cosi_arr:
        for pa_now in pa_arr:

            cos_pa = np.cos(pa_now)
            sin_pa = np.sin(pa_now)
            u_new_d = -cos_pa * u_d + sin_pa *v_d
            v_new_d = -sin_pa * u_d - cos_pa *v_d
            u_new_d = cosi_now * u_new_d
            q_max = np.max(( (u_new_d)**2 + v_new_d**2)**0.5)
            q_max_arr.append(q_max)
    return np.min(q_max_arr)

def main(N_RAD=30, D_PIX= None, DO_TEST_CALC=True, DO_MCMC=True, TAKE_RANDOM_SAMPLE = True, MCMC_RUN=10000, NWALKER=32, \
    DO_SAMPLING=True, SAMPLE_NUM=50, MIN_RSCALE=0.1 * ARCSEC_TO_RAD, MAX_RSCALE=4.0 * ARCSEC_TO_RAD, 
    LOGALPHAMIN=-13, LOGALPHAMAX=-4, LOGALPHAMIN_SAMPLE=-13, LOGALPHAMAX_SAMPLE=-4, D_OFFSET=0.9 * ARCSEC_TO_RAD, NUM_BIN_FOR_DATA_LINEAR=200, NUM_BIN_FOR_DATA_LOG=200, MIN_BIN = 5,  BIN_DATA_REPLACE =True, vis_files = ["./vis_data/test.npz"], outdir ="./out_dir", 
    target_object = None, NUMBER_OF_TARGETS = 1, Q_MAX =None, PA=0, COSI=0, N_process = 4, I_0_input = 0,  START_POS = None, pool = None):
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count_ana_num = 0

    for file in vis_files:

        if NUMBER_OF_TARGETS <= count_ana_num:
            break

        if target_object is not None and target_object != "all":
            if target_object not in file and count_ana_num ==0:
                count_ana_num = 0
                print("OK")
                continue
            else:
                pass
        count_ana_num += 1
        print(file)
        header_name_for_file = file.split("/")[-1].replace(".npz", "")
        u_d, v_d, vis_d, wgt_d = load_obsdata(file)
        q_min_temp = np.min((u_d**2 + v_d**2)**0.5)
        q_max_temp = np.max((u_d**2 + v_d**2)**0.5)
        log10_alpha_mean = 0.5 * (LOGALPHAMIN + LOGALPHAMAX)
        log10_alpha_wd = 0.5 * (-LOGALPHAMIN + LOGALPHAMAX)
        gridfile = os.path.join(outdir, header_name_for_file+ "grid.npz")

  
        if os.path.exists(gridfile) and BIN_DATA_REPLACE==False:
            grid_data = np.load(gridfile)
            u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = grid_data["u"], grid_data["v"], grid_data["vis"], grid_data["noise"]
            q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
            q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)            
        else:
            u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = data_gridding.grid_2dvis_log10_edges_combination_of_two_gridding_using_weights(vis_d, u_d, v_d, wgt_d, NUM_BIN_FOR_DATA_LINEAR, NUM_BIN_FOR_DATA_LOG, q_min_temp, q_max_temp)
            np.savez(gridfile.replace("npz",""), u = u_grid_1d, v = v_grid_1d, vis = vis_grid_1d, noise = noise_grid_1d)
            q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
            q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)
            u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = data_gridding.grid_2dvis_log10_edges_combination_of_two_gridding_using_weights(vis_d, u_d, v_d, wgt_d, NUM_BIN_FOR_DATA_LINEAR, NUM_BIN_FOR_DATA_LOG, q_min, q_max)
        


        if Q_MAX is not None:
            q_d = give_q(u_grid_1d, v_grid_1d, COSI, PA)
            mask = q_d < Q_MAX
            u_grid_1d = u_grid_1d[mask]
            v_grid_1d = v_grid_1d[mask]
            vis_grid_1d = vis_grid_1d[mask]
            noise_grid_1d = noise_grid_1d[mask]

        if D_PIX is None:
            q_max_for_dpix = q_max_determine(u_grid_1d, v_grid_1d)
            D_PIX = 0.5/q_max_for_dpix
        print(D_PIX/ARCSEC_TO_RAD)
        MAXRAD = N_RAD* D_PIX 

        sigma_mat_1d = 1/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
        sigma_mat_1d_for_plot_real = 1/sigma_mat_1d[:int(len(sigma_mat_1d)/2)]**0.5
        sigma_mat_1d_for_plot_imag = 1/sigma_mat_1d[int(len(sigma_mat_1d)/2):]**0.5
        d_data = np.append(vis_grid_1d.real, vis_grid_1d.imag)
        r_n, jn, qmax, q_n = make_collocation_points(MAXRAD, N_RAD)
        q_dist_2d = make_2d_mat_for_dist(q_n)
        N_d, r_dist, d_A_minus1_d, logdet_for_sigma_d = prepare_calculation_for_mcmc(r_n, d_data, sigma_mat_1d)


        if DO_TEST_CALC:
            r_dist = make_2d_mat_for_dist(r_n)
            alpha_test = 10**(-11.5)
            gamma_test = 0.015* ARCSEC_TO_RAD
            #alpha_test = 10**(-11.4)
            #gamma_test = 0.10* ARCSEC_TO_RAD            
            pa_test = 36.5 * np.pi/180 
            inc_test = 0.78
            delta_x_test = 0.0
            delta_y_test = -0.0
            I0_test =0.0000
            #[ 3.92916294e-07  4.16759406e-12  2.87651366e-01  1.02749413e+00, 3.71891044e-07 -3.07274775e-07  1.60470436e-08]
            theta_test = [gamma_test, alpha_test, inc_test, pa_test, delta_x_test, delta_y_test,  I0_test]
            print(theta_test)
            sample_one, H_mat = sample_radial_profile(r_dist, theta_test, u_grid_1d, v_grid_1d, MAXRAD, \
                    N_RAD, D_PIX, d_data, sigma_mat_1d)
            sample_one_forp = sample_one * (ARCSEC_TO_RAD/D_PIX)**2
            plt.plot(r_n/ARCSEC_TO_RAD, sample_one_forp)
            plt.show()

            H_mat, q_dist, d_real_mod, d_imag_mod, vis_model_real, vis_model_imag, u_mod, v_mod= obs_model_comparison(sample_one, u_grid_1d, v_grid_1d, theta_test[2], theta_test[3], theta_test[4], \
                theta_test[5],theta_test[6], d_data, MAXRAD, N_RAD, D_PIX)
            u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist, d_real_mod, 100)
            arg_sort = np.argsort(q_dist)
            plt.plot(q_dist[arg_sort], vis_model_real[arg_sort], lw=1, color="r")
            plt.scatter(u_bin, vis_bin,color="b", s=10)
            plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
            plt.ylim(-0.01, 0.01)
            plt.xscale("log")
            plt.tight_layout()
            plt.show()

        ## emcee

        sample_out_name = os.path.join(outdir, header_name_for_file + "chain_mfreq")
        sample_out_name_npz = sample_out_name + ".npz"

        if DO_MCMC:
            if START_POS  is None:
                delta_pre_fitting = determine_positional_offsets(d_data,  sigma_mat_1d, u_grid_1d, v_grid_1d, n_try = 5000)
            else:
                delta_pre_fitting = np.array(START_POS )

            print(delta_pre_fitting/ARCSEC_TO_RAD)
            initial_for_mcmc = make_initial_geo_offset(LOGALPHAMIN_SAMPLE, LOGALPHAMAX_SAMPLE, MIN_RSCALE, MAX_RSCALE, I_0_input, D_OFFSET*1, NWALKER, delta_pre_fitting[0], delta_pre_fitting[1])
            print(np.shape(initial_for_mcmc))
            if not TAKE_RANDOM_SAMPLE:
                n_iter = 30
                initial_for_minimize = make_initial_geo_offset(LOGALPHAMIN, LOGALPHAMAX, MIN_RSCALE, MAX_RSCALE, I_0_input, D_OFFSET*1, n_iter, delta_pre_fitting[0], delta_pre_fitting[1])
                best_parameters = []
                best_chi = []
                for n_i in range(n_iter): 
                    initial_for_minimize_for_this = initial_for_minimize[n_i]
                    initial_for_minimize_for_this[0] = 0.1 * ARCSEC_TO_RAD
                    initial_for_minimize_for_this[4] = delta_pre_fitting[0]
                    initial_for_minimize_for_this[5] = delta_pre_fitting[1]
                    print(initial_for_minimize_for_this)
                    #min_initial = np.array([np.mean([MIN_RSCALE, MAX_RSCALE]), 10**np.mean([LOGALPHAMIN, LOGALPHAMAX]), 0.5, 0.3 * np.pi, delta_pre_fitting[0], delta_pre_fitting[1] ])
                    res = minimize(log_probability_geo_for_minimizer, initial_for_minimize_for_this, method='L-BFGS-B',\
                            args=(N_d, r_dist,  u_grid_1d, v_grid_1d, \
                    d_data, sigma_mat_1d,  log10_alpha_mean , log10_alpha_wd,  MIN_RSCALE, MAX_RSCALE, MAXRAD, N_RAD, D_PIX), 
                            bounds =( (MIN_RSCALE, MAX_RSCALE), (10**LOGALPHAMIN, 10**LOGALPHAMAX), (0, 1), (0, np.pi), (initial_for_minimize_for_this[4] -0.05 * ARCSEC_TO_RAD,  initial_for_minimize_for_this[4] +0.05 * ARCSEC_TO_RAD), \
                             (initial_for_minimize_for_this[5] -0.05 * ARCSEC_TO_RAD,  initial_for_minimize_for_this[5] +0.05 * ARCSEC_TO_RAD)))
                    #print(res)
                    best_parameters.append(res.x)
                    best_chi.append(res.fun)
                    print(res.x[4]/ARCSEC_TO_RAD, res.x[5]/ARCSEC_TO_RAD)
                    print(res.x, res.fun, res.success)
                best_parameters = np.array(best_parameters)
                best_chi = np.array(best_chi)
                best_arg = np.argmin(best_chi)
                initial_for_mcmc = make_initial_geo_from_optimize(best_parameters[best_arg], LOGALPHAMIN, LOGALPHAMAX, MIN_RSCALE, MAX_RSCALE, D_OFFSET*1,  NWALKER)
            n_w, n_para = np.shape(initial_for_mcmc)
            dtype = [("log_prior", float), ("log_likelihood", float)]
            
            sampler = emcee.EnsembleSampler(NWALKER, n_para, log_probability_geo_for_emcee, args=(N_d, r_dist,  u_grid_1d, v_grid_1d, \
                d_data, sigma_mat_1d,  log10_alpha_mean ,log10_alpha_wd,  MIN_RSCALE, MAX_RSCALE, MAXRAD, N_RAD, D_PIX), blobs_dtype=dtype,  pool=pool)
            sampler.run_mcmc(initial_for_mcmc, MCMC_RUN, progress=True)
            samples = sampler.get_chain(flat=True)
            blobs = sampler.get_blobs()
 
            np.savez(sample_out_name, sample = samples, log_prior =blobs["log_prior"], log_likelihood = blobs["log_likelihood"])

        if DO_SAMPLING:
            n_sample = 3000
            result_mcmc = np.load(sample_out_name_npz)
            sample = result_mcmc["sample"]
            log_likelihood= np.concatenate(result_mcmc["log_likelihood"])
            log_prior= np.concatenate(result_mcmc["log_prior"])
            likeli = np.ravel(result_mcmc["log_likelihood"])
            pos = log_likelihood+ log_prior
            pos_arg = np.argsort(pos)
            sample= sample[pos_arg[len(pos_arg) - n_sample:]]
            n_chain = len(sample)
            plot_out_name_npz = os.path.join(outdir, header_name_for_file +"_mcmc.pdf")
            sample_for_plot_ = samples_for_plot(sample)
            corner.corner(sample_for_plot_)
            plt.tight_layout()
            plt.savefig(plot_out_name_npz, dpi=200)
            plt.close()

            sample_used = sample.T
            nx, nd = np.shape(sample_used)
            index_arr = np.arange(nd)
            sample_num = SAMPLE_NUM
            sample_list = np.array(random.sample(list(index_arr), sample_num))
            sample_mean_arr = []
            spec_arr = []

            for i in range(sample_num):
                theta_now = sample_used[:,sample_list[i]]
                sample_one, H_mat = sample_radial_profile(r_dist, theta_now, u_grid_1d, v_grid_1d, MAXRAD, \
                    N_RAD, D_PIX, d_data, sigma_mat_1d)
                sample_one = sample_one * (ARCSEC_TO_RAD/D_PIX)**2
                plt.plot(r_n/ARCSEC_TO_RAD, sample_one, lw=0.1, color="k")
                sample_mean_arr.append(sample_one)

            sample_mean_arr = np.array(sample_mean_arr)
            sample_mean = np.mean(sample_mean_arr, axis =0)
            plt.plot(r_n/ARCSEC_TO_RAD, sample_mean, lw=1, color="r")
            plt.xlabel("r distance (arcsec)", fontsize = 20)
            plt.ylabel("Flux", fontsize = 20)
            plt.tight_layout()
            np.savez(os.path.join(outdir, header_name_for_file +"radial"), r_pos = r_n/ARCSEC_TO_RAD, 
                sample_arr = sample_mean_arr)
            plt.savefig(os.path.join(outdir, header_name_for_file +"r_dist_flux.png"), dpi=200)
            plt.close()


            theta_test = sample_used[:,sample_list[0]]
            print(sample_for_plot(theta_test))
            H_mat, q_dist_forp, d_real_mod_forp, d_imag_mod_forp, vis_model_real, vis_model_imag, u_mod, v_mod= obs_model_comparison(sample_one, u_grid_1d, v_grid_1d, theta_test[2], theta_test[3], theta_test[4], \
                    theta_test[5], theta_test[6], d_data, MAXRAD, N_RAD, D_PIX) 

            plt.scatter(q_dist_forp, d_real_mod_forp, color="k", s =10)
            plt.errorbar(q_dist_forp, d_real_mod_forp, yerr = sigma_mat_1d_for_plot_real,  ls = "None",  color="k" )
            u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist_forp, d_real_mod_forp, 100)



            for i in range(sample_num):
                theta_test = sample_used[:,sample_list[i]]
                sample_one, H_mat = sample_radial_profile(r_dist, theta_test, u_grid_1d, v_grid_1d, MAXRAD, \
                    N_RAD, D_PIX, d_data, sigma_mat_1d)
                H_mat, q_dist, d_real_mod, d_imag_mod, vis_model_real, vis_model_imag, u_mod, v_mod= obs_model_comparison(sample_one, u_grid_1d, v_grid_1d, theta_test[2], theta_test[3], theta_test[4], \
                    theta_test[5],theta_test[6], d_data, MAXRAD, N_RAD, D_PIX)

                arg_sort = np.argsort(q_dist)
                plt.plot(q_dist[arg_sort], vis_model_real[arg_sort], lw=1, color="r")
            plt.scatter(u_bin, vis_bin,color="b", s=10)
            plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
            plt.tight_layout()

            plt.savefig(os.path.join(outdir, header_name_for_file +"real_comp.png"), dpi=200)
            plt.xscale("log")
            plt.yscale("log")            
            plt.savefig(os.path.join(outdir, header_name_for_file +"real_comp_log.png"), dpi=200)
            plt.close()



            plt.scatter(q_dist_forp, d_imag_mod_forp, color="k", s = 10)
            plt.errorbar(q_dist_forp, d_imag_mod_forp, yerr =  sigma_mat_1d_for_plot_imag, ls = "None",  color="k")
            u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist_forp, d_imag_mod_forp, 100)

            plt.xscale("log")
            for i in range(sample_num):
                theta_test = sample_used[:,sample_list[i]]
                sample_one, H_mat = sample_radial_profile(r_dist, theta_test, u_grid_1d, v_grid_1d, MAXRAD, \
                    N_RAD, D_PIX, d_data, sigma_mat_1d)
                H_mat, q_dist, d_real_mod, d_imag_mod, vis_model_real, vis_model_imag, u_mod, v_mod = obs_model_comparison(sample_one, u_grid_1d, v_grid_1d, theta_test[2], theta_test[3], theta_test[4], \
                    theta_test[5], theta_test[6], d_data, MAXRAD, N_RAD, D_PIX)
                arg_sort = np.argsort(q_dist)
                plt.plot(q_dist[arg_sort], vis_model_imag[arg_sort], lw=1, color="r")
            plt.scatter(u_bin, vis_bin,color="b", s=10)
            plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, header_name_for_file +"imag_comp.png"), dpi=200)
            plt.close()