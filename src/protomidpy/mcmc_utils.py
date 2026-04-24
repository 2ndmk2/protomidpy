import numpy as np
from protomidpy import hankel
ARCSEC_TO_RAD= 1/206265.0


def _effective_pa(pa_profile, weights):
    weights = np.asarray(weights, dtype=float)
    if np.sum(weights) <= 0:
        return float(pa_profile[0])
    comp = np.sum(weights * np.exp(2j * np.asarray(pa_profile)))
    return 0.5 * np.mod(np.angle(comp), 2 * np.pi)


def _warp_output_geometry(I_model, q_dist, cosi_profile, pa_profile):
    weights = np.abs(np.asarray(I_model, dtype=float))
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)
    weights = weights/np.sum(weights)
    q_dist_eff = np.dot(q_dist, weights)
    cosi_eff = float(np.dot(weights, cosi_profile))
    pa_eff = _effective_pa(pa_profile, weights)
    return q_dist_eff, cosi_eff, pa_eff

def obs_model_comparison(I_model, u_d, v_d, theta, d_data, R_out, N, dpix):
    geometry = hankel.geometry_from_theta(theta)
    cosi = geometry["cosi"]
    pa = geometry["pa"]
    delta_x = geometry["delta_x"]
    delta_y = geometry["delta_y"]
    warp_params = geometry["warp"]
    if warp_params is None:
        cos_pa = np.cos(pa)
        sin_pa = np.sin(pa)
        u_new_d_before = -cos_pa * u_d + sin_pa *v_d
        v_new_d = -sin_pa * u_d - cos_pa *v_d
        u_new_d = u_new_d_before * cosi
        q_dist = (u_new_d**2 + v_new_d **2)**0.5
    else:
        r_n, jn, qmax, q_n = hankel.make_collocation_points(R_out, N)
        q_dist_ring, cosi_profile, pa_profile = hankel.make_q_dist_at_inc_pa(
            u_d, v_d, cosi, pa, radii=r_n, warp_params=warp_params, return_profile=True
        )
        q_dist, cosi, pa = _warp_output_geometry(I_model, q_dist_ring, cosi_profile, pa_profile)
        cos_pa = np.cos(pa)
        sin_pa = np.sin(pa)
        u_new_d_before = -cos_pa * u_d + sin_pa *v_d
        v_new_d = -sin_pa * u_d - cos_pa *v_d
        u_new_d = u_new_d_before * cosi
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_cos_inv = np.cos(2 * np.pi * (delta_x * u_d + delta_y * v_d))
    diag_mat_sin_inv = np.sin(2 * np.pi * (delta_x * u_d + delta_y * v_d))
    n_d = int(len(d_data)/2)
    d_real = d_data[:n_d]
    d_imag = d_data[n_d:]
    d_real_mod = d_real * diag_mat_cos_inv  - d_imag * diag_mat_sin_inv
    d_imag_mod = + d_real * diag_mat_sin_inv + d_imag * diag_mat_cos_inv
    if warp_params is None:
        H_mat = hankel.make_hankel_matrix(q_dist, R_out, N, cosi)
    else:
        factor_all, r_pos = hankel.make_hankel_matrix_kataware(R_out, N, dpix)
        H_mat = hankel.make_hankel_at_inc_pa(
            u_d, v_d, geometry["cosi"], geometry["pa"], R_out, N, factor_all, r_pos, dpix, qmax, warp_params=warp_params
        )
    vis_model = np.dot(H_mat, I_model) 
    vis_model_imag = np.zeros(np.shape(vis_model))
    return H_mat, q_dist, d_real_mod, d_imag_mod, vis_model, vis_model_imag, u_new_d_before, v_new_d

def make_model_and_residual(u_d, v_d, theta, I_model, vis_data, R_out, N, dpix):
    geometry = hankel.geometry_from_theta(theta)
    cosi = geometry["cosi"]
    pa = geometry["pa"]
    delta_x = geometry["delta_x"]
    delta_y = geometry["delta_y"]
    warp_params = geometry["warp"]
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d_before = cos_pa * u_d - sin_pa *v_d
    v_new_d = sin_pa * u_d + cos_pa *v_d
    u_new_d = u_new_d_before * cosi
    q_dist = (u_new_d**2 + v_new_d **2)**0.5 
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    if warp_params is None:
        H_mat = hankel.make_hankel_matrix(q_dist, R_out, N, cosi)
    else:
        factor_all, r_pos = hankel.make_hankel_matrix_kataware(R_out, N, dpix)
        r_n, jn, qmax, q_n = hankel.make_collocation_points(R_out, N)
        H_mat = hankel.make_hankel_at_inc_pa(
            u_d, v_d, geometry["cosi"], geometry["pa"], R_out, N, factor_all, r_pos, dpix, qmax, warp_params=warp_params
        )
    vis_model = np.dot(H_mat, I_model) 
    vis_model_real = diag_mat_cos * vis_model 
    vis_model_imag = diag_mat_sin * vis_model 
    vis_model = vis_model_real + 1j*vis_model_imag
    residual = vis_data - vis_model 
    return vis_model, residual 

def make_prior_dic_for_mcmc(para_file):

    file = open(para_file,"r")
    lines = file.readlines()
    para_dic = {}
    for line in lines:
        itemList = line.split()
        para_dic[itemList[0]] = float(itemList[1])
    return para_dic

def make_para_dic_for_mcmc(para_file):
    file = open(para_file,"r")
    lines = file.readlines()
    para_dic = {}
    for line in lines:
        itemList = line.split()
        if itemList[1].isdigit():
            para_dic[itemList[0]] = int(itemList[1])
        else:
            try:
                para_dic[itemList[0]] = float(itemList[1])
            except:
                para_dic[itemList[0]] = str(itemList[1])
    return para_dic


def make_init_mcmc_for_two(para_dic,  nwalker,  x_est = None, y_est = None, target ="", cov="RBF"):
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
    gamma_arr = para_dic["gamma_value"] +  para_dic["gamma_scatter"] * (np.random.rand(nwalker) - 0.5)
    alpha_arr = para_dic["log10_alpha_value"] + para_dic["log10_alpha_scatter"]*(np.random.rand(nwalker) - 0.5)
    para_arr_arr = []
    para_arr_arr.append(gamma_arr) 
    para_arr_arr.append(alpha_arr)
    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr

def make_initial_geo_offset(para_dic,  nwalker,  x_est = None, y_est = None, target ="", cov="RBF"):
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
    gamma_arr = para_dic["gamma_value"] +  para_dic["gamma_scatter"] * (np.random.rand(nwalker) - 0.5)
    alpha_arr = para_dic["log10_alpha_value"] + para_dic["log10_alpha_scatter"]*(np.random.rand(nwalker) - 0.5)
    pa_arr = para_dic["pa_value"] * np.pi/180  + para_dic["pa_scatter"] *(np.random.rand(nwalker) -0.5)* np.pi/180  
    cosi_arr = para_dic["cosi_value"] + para_dic["cosi_scatter"]  *(np.random.rand(nwalker) -0.5) 
    delta_x_arr = para_dic["delta_pos_x"] +   para_dic["delta_pos_scatter"] *(np.random.rand(nwalker) -0.5)
    delta_y_arr =para_dic["delta_pos_y"]  +  para_dic["delta_pos_scatter"]  *(np.random.rand(nwalker) -0.5)    
    if cov=="RBF_double":
        gamma2_arr = para_dic["gamma2_value"] +  para_dic["gamma2_scatter"] * (np.random.rand(nwalker) - 0.5)
    para_arr_arr = []
    para_arr_arr.append(gamma_arr) 
    para_arr_arr.append(alpha_arr)
    para_arr_arr.append(cosi_arr)
    para_arr_arr.append(pa_arr)
    para_arr_arr.append(delta_x_arr)
    para_arr_arr.append(delta_y_arr)
    if "warp_cosi_out_value" in para_dic:
        warp_cosi_out_arr = para_dic["warp_cosi_out_value"] + para_dic["warp_cosi_out_scatter"] * (np.random.rand(nwalker) - 0.5)
        warp_pa_out_arr = para_dic["warp_pa_out_value"] * np.pi/180 + para_dic["warp_pa_out_scatter"] * (np.random.rand(nwalker) - 0.5) * np.pi/180
        warp_r_transition_arr = para_dic["warp_r_transition_value"] + para_dic["warp_r_transition_scatter"] * (np.random.rand(nwalker) - 0.5)
        warp_r_width_arr = para_dic["warp_r_width_value"] + para_dic["warp_r_width_scatter"] * (np.random.rand(nwalker) - 0.5)
        para_arr_arr.append(warp_cosi_out_arr)
        para_arr_arr.append(warp_pa_out_arr)
        para_arr_arr.append(warp_r_transition_arr)
        para_arr_arr.append(warp_r_width_arr)
    if cov=="RBF_double":
        para_arr_arr.append(gamma2_arr)
    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr
