import jax.numpy as np
from georadial_jax import hankel
from georadial_jax import covariance
ARCSEC_TO_RAD= 1/206265.0

def return_log_det(K_cov_inv):
    try:
        (sign, logdet_inv_cov)  = np.linalg.slogdet(K_cov_inv)
        logdet_cov = -logdet_inv_cov
    except:
        logdet_cov = 1e10
    return logdet_cov

def log_prior_geo_two(theta, para_prior_dic):
    """ Compute log prior

    Args:
        theta (ndarray): 1d array for parameters. 
        log10_alpha_mean: (float): log value of mean alpha for prior
        alpha_log_width (float): width of log prior

    Returns:
        log_prior_sum (theta): log prior
    """
    log_prior_sum = 0
    gamma_now = theta[0]
    alpha_now = theta[1]

    if para_prior_dic["min_scale"]  <= gamma_now <= para_prior_dic["max_scale"]:
        log_prior_sum += 0
    else:
        return -np.inf
    if  para_prior_dic["log10_alpha_min"] <= alpha_now <= para_prior_dic["log10_alpha_max"] :
        log_prior_sum += 0
    else:
        return -np.inf

    return log_prior_sum


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


def evidence_for_prob(theta, r_dist,  H_mat, q_dist_model, H_mat_model, d,  sigma_d, cov = "matern",  nu = -100):

    V_A_minus1_U = H_mat.T@hankel.diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*d)
    K_cov, K_cov_inv  = covariance.covariance_return(cov, theta,r_dist, q_dist_model, H_mat_model, H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside) 
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    logdet_cov = return_log_det(K_cov_inv)
    last_term = V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
    log_evidence = - 0.5 * logdet_mat_inside - 0.5 * logdet_cov + 0.5 * last_term
    return log_evidence 

def logp_for_emcee_two(theta, cov, r_dist, para_prior_dic, q_dist_model, H_mat_model, \
    H_mat, V_A_minus1_U, V_A_minus1_d, pa_assumed=0, cosi_assumed=0, delta_x_assumed =0, delta_y_assumed =0):
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

    lp = log_prior_geo_two(theta, para_prior_dic)
    if not np.isfinite(lp):
        return -np.inf, -1e100, -1e100
    theta_new = np.array([theta[0], theta[1]])
    K_cov, K_cov_inv  = covariance.covariance_return(cov, theta_new, r_dist, q_dist_model, H_mat_model, H_mat)
    mat_inside = K_cov_inv + V_A_minus1_U
    mat_inside_inv = np.linalg.inv(mat_inside) 
    (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
    logdet_cov = return_log_det(K_cov_inv)
    last_term = V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
    log_evidence = - 0.5 * logdet_mat_inside - 0.5 * logdet_cov + 0.5 * last_term
    log_pos = lp + log_evidence
    return log_pos,  log_evidence, lp


def test_for_prob_mat_w_fixed_H_mat(other_theta, log10_alpha_arr, gamma_arc_arr,  r_dist, u_d, v_d, vis_d, sigma_d,  R_out, N, dpix, q_dist_model, H_mat_model,  cov="matern", nu = -100):

    evidence_mat = []
    term1_mat = []
    term2_mat = []
    term3_mat = []
    gamma_mat = np.zeros( (len(gamma_arc_arr), len( log10_alpha_arr)))
    alpha_mat = np.zeros( (len(gamma_arc_arr), len( log10_alpha_arr)))
    cosi = other_theta[0]
    pa = other_theta[1]
    delta_x = other_theta[2] *ARCSEC_TO_RAD
    delta_y = other_theta[3] *ARCSEC_TO_RAD

    factor_all, r_pos = hankel.make_hankel_matrix_kataware( R_out, N, dpix)
    q_max_for_dpix = give_q_max(u_d, v_d, other_theta[0], other_theta[1])
    if dpix is None:
        dpix= 0.5/q_max_for_dpix
    r_n, jn, qmax, q_n = hankel.make_collocation_points(R_out, N)
    r_dist = hankel.make_2d_mat_for_dist(r_n)
    H_mat = hankel.make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, factor_all, r_pos, dpix,qmax)
    V_A_minus1_U = H_mat.T@hankel.diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*vis_d)

    for (i, gamma) in enumerate(gamma_arc_arr):
        evidence_arr = []
        term1_arr = []
        term2_arr = []
        term3_arr = []
        for (j,alpha) in enumerate(log10_alpha_arr):
            theta = [gamma, alpha, other_theta[0], other_theta[1], other_theta[2], other_theta[3], other_theta[4]]
            K_cov, K_cov_inv  = covariance.covariance_return(cov, theta,r_dist, q_dist_model, H_mat_model, H_mat)
            mat_inside = K_cov_inv + V_A_minus1_U
            mat_inside_inv = np.linalg.inv(mat_inside) 
            (sign, logdet_mat_inside)  = np.linalg.slogdet(mat_inside)
            logdet_cov = return_log_det(K_cov_inv)
            last_term = V_A_minus1_d.T@ mat_inside_inv @ V_A_minus1_d 
            log_evidence = - 0.5 * logdet_mat_inside - 0.5 * logdet_cov + 0.5 * last_term
            term1_arr.append(logdet_mat_inside)
            term2_arr.append(logdet_cov)
            term3_arr.append(last_term)
            evidence_arr.append(log_evidence)
            gamma_mat[i][j] = gamma
            alpha_mat[i][j] = alpha
        evidence_mat.append(evidence_arr)
        term1_mat.append(term1_arr)
        term2_mat.append(term2_arr)
        term3_mat.append(term3_arr)
        
    return evidence_mat, gamma_mat, alpha_mat, term1_mat, term2_mat, term3_mat

def test_for_prob(theta,  r_dist, u_d, v_d, vis_d, sigma_d,  R_out, N, dpix, q_dist_model, H_mat_model, factor_all, r_pos, cov="matern", nu = -100):
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
    log_evidence , logdet_mat_inside, logdet_cov, last_term = evidence_for_prob(theta, r_dist, H_mat, q_dist_model, H_mat_model,  vis_d, sigma_d, cov = cov,  nu = nu )
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
    log_evidence= evidence_for_prob(theta, r_dist, H_mat, q_dist_model, \
        H_mat_model,  vis_d, sigma_d, cov = cov)
    print(log_evidence, theta)
    log_pos = lp + log_evidence
    return log_pos,  log_evidence, lp

