import os 
import matplotlib.pyplot as plt
import numpy as np
import emcee
from protomidpy import data_gridding
from protomidpy import hankel
from protomidpy import mcmc_utils
from protomidpy import prob
from protomidpy import plotter
import os 

ARCSEC_TO_RAD= 1/206265.0


def grid_search_evidence_core(gamma_arc_arr, log10_alpha_arr, other_theta, N_d, r_dist, u_d, v_d, vis_d, sigma_d,   \
    R_out, N, dpix, q_dist_model, H_mat_model, cov,nu):
    evidence_mat = []
    gamma_mat = np.zeros( (len(gamma_arc_arr), len( log10_alpha_arr)))
    alpha_mat = np.zeros( (len(gamma_arc_arr), len( log10_alpha_arr)))
    max_num = 10.0**(-100)
    term1_mat = []
    term2_mat = []
    term3_mat = []
    factor_all, r_pos = hankel.make_hankel_matrix_kataware( R_out, N, dpix)
    for (i, gamma) in enumerate(gamma_arc_arr):
        evidence_arr = []
        term1_arr = []
        term2_arr = []
        term3_arr = []
        for (j,alpha) in enumerate(log10_alpha_arr):
            theta = [gamma, alpha, other_theta[0], other_theta[1], other_theta[2], other_theta[3], other_theta[4]]
            evidence_now, term1, term2,term3= prob.test_for_prob(theta, r_dist, u_d, v_d, vis_d, sigma_d,  R_out, N, dpix, q_dist_model, H_mat_model, factor_all, r_pos,  cov, nu)
            term1_arr.append(term1)
            term2_arr.append(term2)
            term3_arr.append(term3)
            evidence_arr.append(evidence_now)
            gamma_mat[i][j] = gamma
            alpha_mat[i][j] = alpha
        evidence_mat.append(evidence_arr)
        term1_mat.append(term1_arr)
        term2_mat.append(term2_arr)
        term3_mat.append(term3_arr)
    return evidence_mat, gamma_mat, alpha_mat, term1_mat, term2_mat, term3_mat

def grid_search_evidence_2(gamma_arc_arr, gamma2_arc_arr, log10_alpha_arr, other_theta, N_d, r_dist, u_d, v_d, vis_d, sigma_d,   \
    R_out, N, dpix, q_dist_model, H_mat_model, cov, q_constrained, nu, out_folder =""):
    evidence_mat = []
    gamma_mat = np.zeros( (len(gamma_arc_arr), len( log10_alpha_arr)))
    alpha_mat = np.zeros( (len(gamma_arc_arr), len( log10_alpha_arr)))
    max_num = 10.0**(-100)
    term1_mat = []
    term2_mat = []
    term3_mat = []
    factor_all, r_pos = hankel.make_hankel_matrix_kataware( R_out, N, dpix)

    for (i, gamma) in enumerate(gamma_arc_arr):
        evidence_arr = []
        term1_arr = []
        term2_arr = []
        term3_arr = []
        for (j,alpha) in enumerate(log10_alpha_arr):
            for (k,gamma2) in enumerate(gamma2_arc_arr):
                theta = [gamma, alpha, other_theta[0], other_theta[1], other_theta[2], other_theta[3], other_theta[4], gamma2]
                evidence_now, term1, term2,term3= prob.test_for_prob(theta, r_dist, u_d, v_d, vis_d, sigma_d,  R_out, N, dpix, q_dist_model, H_mat_model, factor_all, r_pos,  cov, q_constrained, nu)
                term1_arr.append(term1)
                term2_arr.append(term2)
                term3_arr.append(term3)
                evidence_arr.append(evidence_now)
                gamma_mat[i][j] = gamma
                alpha_mat[i][j] = alpha
                #print(gamma, alpha, evidence_now)
                if evidence_now >max_num:
                    max_num = evidence_now 
                    print(gamma, alpha, evidence_now)
            evidence_mat.append(evidence_arr)
            term1_mat.append(term1_arr)
            term2_mat.append(term2_arr)
            term3_mat.append(term3_arr)
    evidence_mat = np.array(evidence_mat)

    return evidence_mat, gamma_mat, alpha_mat, term1_mat, term2_mat, term3_mat



def grid_search_evidence_bin_rad(u_d, v_d, vis_d, wgt_d, cov, nu_now, log10_alpha_arr, gamma_arc_arr,  header_name_for_file = "test", out_dir = "./", nrad=300, dpix= 0.1 * ARCSEC_TO_RAD, 
    n_bin_log=200,  pa_assumed=0, q_min_max_bin = [1e3, 1e7], cosi_assumed=0, delta_x_assumed =0, delta_y_assumed =0):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    R_out = nrad* dpix 
    other_thetas = [cosi_assumed,  pa_assumed, delta_x_assumed, delta_y_assumed, 0]
    gridfile = os.path.join(out_dir, header_name_for_file) +"grid"
    q, q_v_zero , vis_d, wgt_d, noise_d, sigma_mat_1d, d_data = data_gridding.deproject_radial(u_d, v_d, vis_d, wgt_d, cosi_assumed, pa_assumed , delta_x_assumed, delta_y_assumed)  
    coord_for_grid  = data_gridding.log_gridding_1d( q_min_max_bin[0], q_min_max_bin[1],n_bin_log)
    q_grid_1d_lg_data, vis_grid_1d_lg_data, noise_grid_1d_lg_data, d_data_grid_1d, sigma_mat_grid_1d = data_gridding.data_binning_1d(q, vis_d, wgt_d,coord_for_grid)
    np.savez(gridfile, u = q_grid_1d_lg_data, v = np.zeros_like(q_grid_1d_lg_data), vis = vis_grid_1d_lg_data, noise = noise_grid_1d_lg_data)
    r_n, jn, qmax, q_n, H_mat_model, q_dist_2d_model, N_d, r_dist, d_A_minus1_d, logdet_for_sigma_d  = hankel.prepare(R_out, nrad,  d_data_grid_1d, sigma_mat_grid_1d)
    u_grid_1d, v_grid_1d= q_grid_1d_lg_data, np.zeros_like(q_grid_1d_lg_data)
    H_mat , V_A_minus1_U , V_A_minus1_d = hankel.hankel_precomp(R_out, nrad, dpix, u_grid_1d,  v_grid_1d, d_data_grid_1d, sigma_mat_grid_1d, 1.0,0,0,0)
    for log10_alpha in log10_alpha_arr:
        for gamma_arc in gamma_arc_arr:
            gamma_input = [gamma_arc]
            plotter.sample_and_gallary(out_dir, log10_alpha,gamma_input, r_n, other_thetas, u_grid_1d, v_grid_1d, R_out, nrad, dpix, d_data_grid_1d, \
                sigma_mat_grid_1d,  q_dist_2d_model, H_mat_model, cov , H_mat , V_A_minus1_U , V_A_minus1_d )
    evidence_mat, gamma_mat, alpha_mat, term1_mat, term2_mat, term3_mat = prob.test_for_prob_mat_w_fixed_H_mat(other_thetas, log10_alpha_arr, gamma_arc_arr,  r_dist, \
        u_grid_1d, v_grid_1d, d_data_grid_1d, sigma_mat_grid_1d, R_out,nrad, dpix, q_dist_2d_model, H_mat_model, cov, nu_now )
    np.savez(os.path.join(out_dir, "evidence") , evidence_mat = evidence_mat, gamma_mat = gamma_mat, alpha_mat = alpha_mat, \
            term1_mat = term1_mat, term2_mat = term2_mat, term3_mat = term3_mat, log_alpha_arr =  log10_alpha_arr, gamma_arr = gamma_arc_arr)

    return q_grid_1d_lg_data, vis_grid_1d_lg_data, noise_grid_1d_lg_data, d_data_grid_1d, sigma_mat_grid_1d

def grid_search_evidence(u_d, v_d, vis_d, wgt_d, cov, nu_now, log10_alpha_arr, gamma_arc_arr,  header_name_for_file = "test", out_dir = "./", nrad=30, dpix= 0.1 * ARCSEC_TO_RAD, 
    n_bin_linear=200, n_bin_log=200,  q_min_max_bin = [1e3, 1e7], pa_assumed=0, cosi_assumed=0, delta_x_assumed =0, delta_y_assumed =0):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    R_out = nrad* dpix 
    other_thetas = [cosi_assumed,  pa_assumed, delta_x_assumed, delta_y_assumed, 0]
    gridfile = os.path.join(out_dir, header_name_for_file+ "grid.npz")
    u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d, sigma_mat_1d, d_data = data_gridding.get_gridded_obs_data(gridfile, u_d, v_d, vis_d, wgt_d, n_bin_linear, n_bin_log, q_min_max_bin)
    r_n, jn, qmax, q_n, H_mat_model, q_dist_2d_model, N_d, r_dist, d_A_minus1_d, logdet_for_sigma_d  = hankel.prepare(R_out, nrad,  d_data, sigma_mat_1d)

    H_mat , V_A_minus1_U , V_A_minus1_d = hankel.hankel_precomp(R_out, nrad, dpix, u_grid_1d, v_grid_1d, d_data, sigma_mat_1d, cosi_assumed, pa_assumed,  delta_x_assumed, delta_y_assumed)
    for log10_alpha in log10_alpha_arr:
        for gamma_arc in gamma_arc_arr:
            gamma_input = [gamma_arc]
            plotter.sample_and_gallary(out_dir, log10_alpha,gamma_input, r_n, other_thetas, u_grid_1d, v_grid_1d, R_out, nrad, dpix, d_data, \
                sigma_mat_1d,  q_dist_2d_model, H_mat_model, cov , H_mat , V_A_minus1_U , V_A_minus1_d )
    evidence_mat, gamma_mat, alpha_mat, term1_mat, term2_mat, term3_mat = prob.test_for_prob_mat_w_fixed_H_mat(other_thetas, log10_alpha_arr, gamma_arc_arr,  r_dist, \
        u_grid_1d, v_grid_1d, d_data, sigma_mat_1d, R_out,nrad, dpix, q_dist_2d_model, H_mat_model, cov, nu_now )
    np.savez(os.path.join(out_dir, "evidence") , evidence_mat = evidence_mat, gamma_mat = gamma_mat, alpha_mat = alpha_mat, \
            term1_mat = term1_mat, term2_mat = term2_mat, term3_mat = term3_mat, log_alpha_arr =  log10_alpha_arr, gamma_arr = gamma_arc_arr)


def data_binning(u_d, v_d, vis_d, wgt_d, cov, nu_now,  n_walker, n_chain, para_dic_for_prior, para_dic_for_mcmc, header_name_for_file = "test", out_dir = "./", nrad=300, dpix= 0.1 * ARCSEC_TO_RAD, 
    n_bin_log=200,  q_min_max_bin = [1e3, 1e7], file_for_prior = "", file_for_mcmc = "", pool =None):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    coord_for_grid_lg, rep_positions_for_grid_lg, uu_for_grid_pos_lg, vv_for_grid_pos_lg = data_gridding.log_gridding_2d(q_min_max_bin[0], q_min_max_bin[1], n_bin_log)
    u_grid_2d, v_grid_2d, vis_grid_2d, noise_grid_2d, sigma_mat_2d, d_data,  binnumber = \
        data_gridding.data_binning_2d(u_d, v_d,vis_d, wgt_d, coord_for_grid_lg)
    gridfile = os.path.join(out_dir, header_name_for_file+ "_grid.npz")
    np.savez(gridfile, u = u_grid_2d, v = v_grid_2d, vis = vis_grid_2d, noise = noise_grid_2d)
  
def after_sampling(u, v, mcmc_para, nrad=300, dpix= 0.1 * ARCSEC_TO_RAD):
    mcmc_result = np.load(mcmc_para)
    sample = mcmc_result["sample"]
    log_posterior = mcmc_result["log_prior"] + mcmc_result["log_likelihood"] 
    sample_best = sample[:,np.argmax(log_posterior)]


def sample_mcmc_full(u_d, v_d, vis_d, wgt_d, cov, nu_now,  n_walker, n_chain, para_dic_for_prior, para_dic_for_mcmc, header_name_for_file = "test", out_dir = "./", nrad=300, dpix= 0.1 * ARCSEC_TO_RAD, 
    n_bin_log=200,  q_min_max_bin = [1e3, 1e7], pool =None):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    coord_for_grid_lg, rep_positions_for_grid_lg, uu_for_grid_pos_lg, vv_for_grid_pos_lg = data_gridding.log_gridding_2d(q_min_max_bin[0], q_min_max_bin[1], n_bin_log)
    u_grid_2d, v_grid_2d, vis_grid_2d, noise_grid_2d, sigma_mat_2d, d_data,  binnumber = \
        data_gridding.data_binning_2d(u_d, v_d,vis_d, wgt_d, coord_for_grid_lg)
    gridfile = os.path.join(out_dir, header_name_for_file+ "_grid.npz")
    np.savez(gridfile, u = u_grid_2d, v = v_grid_2d, vis = vis_grid_2d, noise = noise_grid_2d)
    R_out = nrad* dpix 
    r_n, jn, qmax, q_n, H_mat_model, q_dist_2d_model, N_d, r_dist, d_A_minus1_d, logdet_for_sigma_d  = hankel.prepare(R_out, nrad,  d_data, sigma_mat_2d)

    ## emcee
    initial_for_mcmc = mcmc_utils.make_initial_geo_offset(para_dic_for_mcmc, n_walker, cov = cov)
    n_w, n_para = np.shape(initial_for_mcmc)
    dtype = [ ("log_likelihood", float), ("log_prior", float)]
    factor_all, r_pos= hankel.make_hankel_matrix_kataware( R_out, nrad, dpix)
    sampler = emcee.EnsembleSampler(n_walker, n_para, prob.log_probability_geo_for_emcee, args=(N_d, r_dist,  u_grid_2d, v_grid_2d, \
        d_data, sigma_mat_2d,  para_dic_for_prior, R_out, nrad, dpix, q_dist_2d_model, H_mat_model, factor_all, r_pos, cov), blobs_dtype=dtype,  pool=pool)
    sampler.run_mcmc(initial_for_mcmc, n_chain, progress=True)
    samples = sampler.get_chain(flat=True)
    blobs = sampler.get_blobs()
    sample_out_name = os.path.join(out_dir, header_name_for_file +"_mcmc")
    np.savez(sample_out_name, sample = samples, log_prior =blobs["log_prior"], \
        log_likelihood = blobs["log_likelihood"], nrad= nrad, dpix=dpix/ARCSEC_TO_RAD, \
        n_bin_log =n_bin_log, qmin=  q_min_max_bin[0], qmax =  q_min_max_bin[1], cov =cov)
    
    ## corner plot
    mcmc_plot = True
    if mcmc_plot:
        plotter.mcmc_plot(sample_out_name + ".npz", out_dir, header_name_for_file)


