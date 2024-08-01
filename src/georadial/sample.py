import numpy as np
from protomidpy import hankel
from protomidpy import covariance
ARCSEC_TO_RAD= 1/206265.0


def sample_radial_profile_fixed_geo(cov, theta,r_dist, q_dist_model, H_mat_model, V_A_minus1_U ,V_A_minus1_d, H_mat):
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
    K_cov, K_cov_inv  = covariance.covariance_return(cov, theta,r_dist, q_dist_model, H_mat_model, H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U    
    mat_inside_inv = np.linalg.inv(mat_inside)
    mean_gaussian =  np.dot(mat_inside_inv, V_A_minus1_d)
    sample_one = np.random.multivariate_normal(mean_gaussian, mat_inside_inv)
    return sample_one


def sample_radial_profile(r_dist, theta, u_d, v_d, R_out, N, dpix,  d, sigma_d, q_dist_model, H_mat_model, cov="matern", nu = -100 ):
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
    #print(R_out, N, dpix)
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]* ARCSEC_TO_RAD
    delta_y = theta[5]* ARCSEC_TO_RAD
    r_n, jn, qmax, q_n = hankel.make_collocation_points(R_out, N)
    factor_all, r_pos = hankel.make_hankel_matrix_kataware( R_out, N, dpix)
    H_mat = hankel.make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, factor_all, r_pos, dpix,  qmax)
    V_A_minus1_U = H_mat.T@hankel.diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*d)
    K_cov, K_cov_inv  = covariance.covariance_return(cov, theta,r_dist, q_dist_model, H_mat_model, H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U    
    mat_inside_inv = np.linalg.inv(mat_inside)
    mean_gaussian =  np.dot(mat_inside_inv, V_A_minus1_d)
    sample_one = np.random.multivariate_normal(mean_gaussian, mat_inside_inv)
    return sample_one, H_mat
