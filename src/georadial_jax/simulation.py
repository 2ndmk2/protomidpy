import numpy as np
from georadial import hankel
from georadial import covariance

def log_prior_geo(theta, para_prior_dic):
    """ Compute log prior

    Args:
        theta (ndarray): 1d array for parameters. 
        log10_alpha_mean: (float): log value of mean alpha for prior
        alpha_log_width (float): width of log prior

    Returns:
        log_prior_sum (theta): log prior
    """
    log_prior_sum = 0
    if len(theta)==7:
        gamma_arr = [theta[0], theta[6]]
        if theta[0] > theta[6]:
            return -np.inf
    else:
        gamma_arr = [theta[0]]
    alpha_arr = [theta[1]]
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    arcsecond = 1/206265.0
    for gamma_now in gamma_arr:
        if para_prior_dic["min_scale"]  <= gamma_now <= para_prior_dic["max_scale"]:
            log_prior_sum += 0
        else:
            return -np.inf
    for alpha_now in alpha_arr:
        if  para_prior_dic["log10_alpha_min"] <= alpha_now <= para_prior_dic["log10_alpha_max"] :
            log_prior_sum += 0
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

    if -para_prior_dic["delta_pos"]<= delta_x <= para_prior_dic["delta_pos"] :
        log_prior_sum += 0
    else:
        return -np.inf

    if -para_prior_dic["delta_pos"]  <= delta_y <= para_prior_dic["delta_pos"] :
        log_prior_sum += 0
    else:
        return -np.inf

    return log_prior_sum


def evidence_for_prob(theta, r_dist,  H_mat, q_dist_model, H_mat_model, d,  sigma_d, cov = "matern", q_constrained = 0, nu = -100):


    V_A_minus1_U = H_mat.T@hankel.diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*d)
    K_cov, K_cov_inv  = covariance.covariance_return(cov, theta,r_dist, q_dist_model, H_mat_model, H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside) 
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    try:
        (sign, logdet_inv_cov)  = np.linalg.slogdet(K_cov_inv)
        logdet_cov = -logdet_inv_cov
    except:
        logdet_cov = 1e10
    last_term = V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
    log_evidence = - 0.5 * logdet_mat_inside - 0.5 * logdet_cov + 0.5 * last_term
    return log_evidence , - 0.5 * logdet_mat_inside, - 0.5 * logdet_cov, 0.5 * last_term


def test_for_prob(theta,  r_dist, u_d, v_d, vis_d, sigma_d,  R_out, N, dpix, q_dist_model, H_mat_model, factor_all, r_pos, cov="matern", q_constrained = 0, nu = -100):
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
    r_n, jn, qmax, q_n = hankel.make_collocation_points(R_out, N)
    r_dist = hankel.make_2d_mat_for_dist(r_n)
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4] *ARCSEC_TO_RAD
    delta_y = theta[5] *ARCSEC_TO_RAD
    H_mat = hankel.make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, factor_all, r_pos, dpix,qmax)
    log_evidence , logdet_mat_inside, logdet_cov, last_term = evidence_for_prob(theta, r_dist, H_mat, q_dist_model, H_mat_model,  vis_d, sigma_d, cov = cov, q_constrained = q_constrained, nu = nu )
    return log_evidence , logdet_mat_inside, logdet_cov, last_term

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

def log_probability_geo_for_emcee(theta, N_d, r_dist, u_d, v_d, vis_d, sigma_d,  para_prior_dic, R_out, N, dpix, q_dist_model, H_mat_model, factor_all, r_pos, cov="matern"):
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

    lp = log_prior_geo(theta, para_prior_dic)
    if not np.isfinite(lp):
        return -np.inf, -1e100, -1e100
    q_max_for_dpix = give_q_max(u_d, v_d, theta[2], theta[3])
    if dpix is None:
        dpix= 0.5/q_max_for_dpix
    R_out = N * dpix    
    r_n, jn, qmax, q_n = hankel.make_collocation_points(R_out, N)
    r_dist = hankel.make_2d_mat_for_dist(r_n)
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4] * ARCSEC_TO_RAD
    delta_y = theta[5]* ARCSEC_TO_RAD
    H_mat = hankel.make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, factor_all, r_pos,  dpix, qmax)
    log_evidence , test1, test2, test3 = evidence_for_prob(theta, r_dist, H_mat, q_dist_model, \
        H_mat_model,  vis_d, sigma_d, cov = cov)
    print(log_evidence, theta)
    log_pos = lp + log_evidence
    return log_pos,  log_evidence, lp

