import numpy as np
import matplotlib.pyplot as plt
import glob
from protomidpy import data_gridding
from protomidpy import sample
from protomidpy import utils
from protomidpy import mcmc_utils
from protomidpy import hankel
import os
ARCSEC_TO_RAD= 1/206265.0

## Change here if needed!!!
n_sample_for_rad = 20
n_burnin = 20000 
mcmc_result_file = "./result/AS209_continuum_averaged.vis_mcmc.npz"
visfile = "./vis_data/AS209_continuum_averaged.vis.npz" 
out_file_for_model = "./result/AS209_continuum_averagedmodel.npz"

###

mcmc_result = np.load(mcmc_result_file)
sample_now = mcmc_result["sample"]
log_posterior = mcmc_result["log_prior"] + mcmc_result["log_likelihood"] 
sample_goods = sample_now[n_burnin:,:]
sample_best = sample_now[np.argmax(np.ravel(log_posterior)),:]
n_bin_log = mcmc_result["n_bin_log"]
nrad = mcmc_result["nrad"] 
dpix= mcmc_result["dpix"] * ARCSEC_TO_RAD
cov = mcmc_result["cov"]
R_out = nrad * dpix
q_min_max_bin = [mcmc_result["qmin"], mcmc_result["qmax"]]

u_d, v_d, vis_d, wgt_d, freq_d = utils.load_obsdata(visfile)
data_d = np.append(vis_d.real, vis_d.imag)
coord_for_grid_lg, rep_positions_for_grid_lg, uu_for_grid_pos_lg, vv_for_grid_pos_lg = data_gridding.log_gridding_2d(q_min_max_bin[0], q_min_max_bin[1], n_bin_log)
u_grid_2d, v_grid_2d, vis_grid_2d, noise_grid_2d, sigma_mat_2d, d_data,  binnumber = \
    data_gridding.data_binning_2d(u_d, v_d,vis_d, wgt_d, coord_for_grid_lg)
r_n, jn, qmax, q_n, H_mat_model, q_dist_2d_model, N_d, r_dist, d_A_minus1_d, logdet_for_sigma_d  = hankel.prepare(R_out, nrad,  d_data, sigma_mat_2d)

flux_arr = []
sample_random_selected = sample_goods[np.random.choice(np.arange(len(sample_goods)),n_sample_for_rad)]
for sample_now in sample_goods[:n_sample_for_rad,:]:
    flux_sampled, H_mat = sample.sample_radial_profile(r_dist, sample_now, u_grid_2d, v_grid_2d, R_out, \
                nrad, dpix, d_data, sigma_mat_2d, q_dist_2d_model, H_mat_model, cov=cov)
    flux_arr.append(flux_sampled)

sample_one_taken, H_mat = sample.sample_radial_profile(r_dist, sample_best, u_grid_2d, v_grid_2d, R_out, \
                nrad, dpix, d_data, sigma_mat_2d, q_dist_2d_model, H_mat_model, cov=cov)
H_mat, q_dist, d_real_mod, d_imag_mod, vis_model_real, vis_model_imag, u_mod, v_mod= mcmc_utils.obs_model_comparison(sample_one_taken, u_d, v_d, sample_best, data_d , R_out, nrad, dpix)
vis_model, residual  = mcmc_utils.make_model_and_residual(u_d, v_d, sample_best, sample_one_taken, vis_d, R_out, nrad, dpix)
np.savez(out_file_for_model, r_n= r_n, param_map = sample_best, params_random_selected= sample_random_selected, flux_map_sample = sample_one_taken, flux_random_samples = flux_arr, 
    vis_model_undeprojected = vis_model, residual_undeprojected = residual, qdist_deprojected = q_dist, vis_model_deprojected = vis_model_real+1j*vis_model_imag, data_deprojected = d_real_mod +1j * d_imag_mod, data_weights = wgt_d)

