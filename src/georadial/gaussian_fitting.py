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
#from georadial.geo_fitting import * 
from multiprocessing import Pool, freeze_support

ARCSEC_TO_RAD= 1/206265.0
os.environ["OMP_NUM_THREADS"] = "1"

def prior_for_gauss(theta, delta_pos = 1):
    arcsecond = 1/206265.0
    I_0 = theta[0]
    R_out = theta[1]
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    log_prior_sum = 0

    if  0 < I_0:
        log_prior_sum += 0
    else:
        return -np.inf

    if  0 < R_out < 2*ARCSEC_TO_RAD:
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

    if -delta_pos* arcsecond <= delta_x <= delta_pos * arcsecond:
        log_prior_sum += 0
    else:
        return -np.inf

    if -delta_pos * arcsecond <= delta_y <= delta_pos * arcsecond:
        log_prior_sum += 0
    else:
        return -np.inf

    return log_prior_sum

def log_probability_gaussian_for_minimizer(theta, u_d, v_d, vis_d, sigma_d ):
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
   
    I_0 = theta[0]
    R_out = theta[1]
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]

    lp = prior_for_gauss(theta)
    if not np.isfinite(lp):
        return -np.inf

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d = -cos_pa * u_d + sin_pa *v_d
    v_new_d = -sin_pa * u_d - cos_pa *v_d
    u_new_d = u_new_d * cosi
    q_dist = (u_new_d**2 + v_new_d **2)**0.5
    model_gauss = I_0 * np.exp(-2 * (np.pi**2) * (R_out**2) * (q_dist**2))
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat = np.append(diag_mat_cos * model_gauss, diag_mat_sin * model_gauss)
    lhood = - 0.5 *  np.sum(sigma_d * (diag_mat - vis_d) **2 )
    log_pos = lp +lhood
    return log_pos



def log_probability_gaussian_for_emcee(theta, u_d, v_d, vis_d, sigma_d ):
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
   
    I_0 = theta[0]
    R_out = theta[1]
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]

    lp = prior_for_gauss(theta)
    if not np.isfinite(lp):
        return -np.inf, -10**6, -10**6, -10**6

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d = -cos_pa * u_d + sin_pa *v_d
    v_new_d = -sin_pa * u_d - cos_pa *v_d
    u_new_d = u_new_d * cosi
    q_dist = (u_new_d**2 + v_new_d **2)**0.5
    model_gauss = I_0 * np.exp(-2 * (np.pi**2) * (R_out**2) * (q_dist**2))
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat = np.append(diag_mat_cos * model_gauss, diag_mat_sin * model_gauss)
    lhood = - 0.5 *  np.sum(sigma_d * (diag_mat - vis_d) **2 )
    log_pos = lp +lhood
    chi = -2  * lhood
    return log_pos, lp, lhood, chi



def make_initial_gauss(Imin, Imax, rmin, rmax, x_est = None, y_est = None, nwalker = 32, delta_pos = 1 * ARCSEC_TO_RAD):

    I_arr  =  Imin + np.random.rand(nwalker) * (Imax - Imin)
    R_out_arr  =  rmin + np.random.rand(nwalker) * (rmax - rmin)
    pa_arr = np.random.rand(nwalker) * np.pi  
    cosi_arr = np.random.rand(nwalker) 
    if x_est is not None:
        delta_x_arr = x_est +  0.001  *  np.random.rand(nwalker) * delta_pos
        delta_y_arr = y_est  +  0.001  *  np.random.rand(nwalker) * delta_pos
    else:
        delta_x_arr = 0.0 * ARCSEC_TO_RAD  + 0.001 *  np.random.rand(nwalker) * delta_pos
        delta_y_arr = 0.0 * ARCSEC_TO_RAD  +  0.001  *  np.random.rand(nwalker) * delta_pos

    para_arr_arr = []
    para_arr_arr.append(I_arr)
    para_arr_arr.append(R_out_arr)
    para_arr_arr.append(cosi_arr)
    para_arr_arr.append(pa_arr)
    para_arr_arr.append(delta_x_arr)
    para_arr_arr.append(delta_y_arr)
    para_arr_arr = np.array(para_arr_arr)
    para_arr_arr = para_arr_arr.T
    return para_arr_arr


def samples_for_plot_gaussian(samples):


    samples_update = np.copy(samples)
    samples_update[:,0] = samples[:,0]
    samples_update[:,3] = samples[:,3] * 180/np.pi
    samples_update[:,1] = samples[:,1]/ARCSEC_TO_RAD
    samples_update[:,2] = samples[:,2] 
    samples_update[:,4] = samples[:,4]/ARCSEC_TO_RAD
    samples_update[:,5] = samples[:,5]/ARCSEC_TO_RAD
    return samples_update

def sample_gaussian(theta, rmax = 3):

    I_0 = theta[0]
    R_out = theta[1]
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    r_pos = np.linspace(0, rmax * ARCSEC_TO_RAD, 1000)
    flux = I_0 * (1/(2 * (R_out**2) * np.pi)) * np.exp(-r_pos**2/(2 *R_out**2 ))
    flux_arcsec2 = flux * ARCSEC_TO_RAD**2

    return r_pos, flux, flux_arcsec2

def sample_gaussian_for_qspace(theta, qmax = 3):

    I_0 = theta[0]
    R_out = theta[1]
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]
    q_pos = np.linspace(0, qmax, 100)
    vis_model = I_0 * np.exp(-2 * (np.pi**2) * (R_out**2) * (q_pos**2))

    return q_pos,vis_model 



def gaussian_model_comparison(u_d, v_d, d_data, theta):

    I_0 = theta[0]
    R_out = theta[1]
    cosi = theta[2]
    pa = theta[3]
    delta_x = theta[4]
    delta_y = theta[5]

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
    vis_model = I_0 * np.exp(-2 * (np.pi**2) * (R_out**2) * (q_dist**2))
    vis_model_imag = np.zeros(np.shape(vis_model))

    return q_dist, d_real_mod, d_imag_mod, vis_model, vis_model_imag



def main_gauss(DO_MCMC=True, TAKE_RANDOM_SAMPLE = True, MCMC_RUN=10000, NWALKER=32, \
    DO_SAMPLING=True, SAMPLE_NUM=50, NUM_BIN_FOR_DATA_LINEAR=200, NUM_BIN_FOR_DATA_LOG=200, MIN_BIN = 5,  \
    BIN_DATA_REPLACE =True, vis_folder = "./vis_data", outdir ="./out_dir_gauss", 
    target_object = None, NUMBER_OF_TARGETS = 1, N_process = 4):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count_ana_num = 0
    files = glob.glob(vis_folder + "/*")
    for file in files:

        if NUMBER_OF_TARGETS <= count_ana_num:
            break

        if target_object is not None and target_object != "all":


            if target_object not in file and count_ana_num ==0:
                count_ana_num = 0
                continue
            else:
                pass
        count_ana_num += 1
        header_name_for_file = file.split("/")[-1].replace(".npz", "")
        u_d, v_d,  vis_d = load_obsdata(file)
        q_min_temp = np.min((u_d**2 + v_d**2)**0.5)
        q_max_temp = np.max((u_d**2 + v_d**2)**0.5)
        gridfile = os.path.join(outdir, header_name_for_file+ "grid.npz")

        if os.path.exists(gridfile) and BIN_DATA_REPLACE==False:
            grid_data = np.load(gridfile)
            u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d, count_grid_1d = grid_data["u"], grid_data["v"], grid_data["vis"],\
            grid_data["noise"], grid_data["count"]
            noise_grid_1d = np.mean(noise_grid_1d * np.sqrt(count_grid_1d))/np.sqrt(count_grid_1d)
            q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
            q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)            
        else:
            u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d, count_grid_1d = data_gridding.grid_2dvis_log10_edges_combination_of_two_gridding(vis_d, u_d, v_d, NUM_BIN_FOR_DATA_LINEAR, NUM_BIN_FOR_DATA_LOG, q_min_temp, q_max_temp, MIN_BIN)
            np.savez(gridfile.replace("npz",""), u = u_grid_1d, v = v_grid_1d, vis = vis_grid_1d, noise = noise_grid_1d, count = count_grid_1d)
            noise_grid_1d = np.mean(noise_grid_1d * np.sqrt(count_grid_1d))/np.sqrt(count_grid_1d)
            q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
            q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)
            u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d, count_grid_1d = data_gridding.grid_2dvis_log10_edges_combination_of_two_gridding(vis_d, u_d, v_d, NUM_BIN_FOR_DATA_LINEAR, NUM_BIN_FOR_DATA_LOG, q_min, q_max, MIN_BIN)
            noise_grid_1d = np.mean(noise_grid_1d * np.sqrt(count_grid_1d))/np.sqrt(count_grid_1d)

        sigma_mat_1d = 1/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
        sigma_mat_1d_for_plot_real = 1/sigma_mat_1d[:int(len(sigma_mat_1d)/2)]**0.5
        sigma_mat_1d_for_plot_imag = 1/sigma_mat_1d[int(len(sigma_mat_1d)/2):]**0.5
        d_data = np.append(vis_grid_1d.real, vis_grid_1d.imag)

        sample_out_name = os.path.join(outdir, header_name_for_file + "chain_mfreq")
        sample_out_name_npz = sample_out_name + ".npz"

        if DO_MCMC:
            delta_pre_fitting = determine_positional_offsets(d_data,  sigma_mat_1d, u_grid_1d, v_grid_1d, n_try = 500)
            initial_for_mcmc = make_initial_gauss(0.001, 1, 0.01*ARCSEC_TO_RAD, 1 * ARCSEC_TO_RAD, \
                delta_pre_fitting[0], delta_pre_fitting[1], NWALKER)
            n_w, n_para = np.shape(initial_for_mcmc)
            dtype = [("log_prior", float), ("log_likelihood", float), ("chi_sq", float)]
            with Pool(processes=N_process) as pool:
                sampler = emcee.EnsembleSampler(NWALKER, n_para, log_probability_gaussian_for_emcee, args=(u_grid_1d, v_grid_1d, \
                d_data, sigma_mat_1d), blobs_dtype=dtype,  pool=pool)
                sampler.run_mcmc(initial_for_mcmc, MCMC_RUN, progress=True)
            samples = sampler.get_chain(flat=True)
            blobs = sampler.get_blobs()
 
            np.savez(sample_out_name, sample = samples, log_prior =blobs["log_prior"], log_likelihood = blobs["log_likelihood"],\
             chi_sq = blobs["chi_sq"])

        if DO_SAMPLING:
            result_mcmc = np.load(sample_out_name_npz)
            sample = result_mcmc["sample"]
            chi_sq = np.ravel(result_mcmc["chi_sq"])
            lp = result_mcmc["log_prior"]# ("log_likelihood
            likeli = result_mcmc["log_likelihood"]
            post = np.ravel(lp) + np.ravel(likeli)
            mask = (sample[:,1]>0. * ARCSEC_TO_RAD) * (chi_sq>0) #* ((post>np.max(post)-20))
            sample = sample[mask]
            n_chain = len(sample)
            burn_out = n_chain - 10000
            plot_out_name_npz = os.path.join(outdir, header_name_for_file +"_mcmc.pdf")
            sample_for_plot_ = samples_for_plot_gaussian(sample[burn_out:])
            corner.corner(sample_for_plot_)
            plt.tight_layout()
            plt.savefig(plot_out_name_npz, dpi=200)
            plt.close()

            sample = np.load(sample_out_name_npz)["sample"][mask].T
            sample_used = sample[:,burn_out:]
            nx, nd = np.shape(sample_used)
            index_arr = np.arange(nd)
            sample_num = SAMPLE_NUM
            sample_list = np.array(random.sample(list(index_arr), sample_num))
            sample_mean_arr = []
            spec_arr = []

            ## flux
            for i in range(sample_num):
                theta_now = sample_used[:,sample_list[i]]
                r_pos, flux_one, flux_arcsec2 = sample_gaussian(theta_now)
                plt.plot(r_pos/ARCSEC_TO_RAD, flux_arcsec2, lw=0.1, color="k")
                sample_mean_arr.append(flux_arcsec2)
            sample_mean_arr = np.array(sample_mean_arr)
            sample_mean = np.mean(sample_mean_arr, axis =0)
            plt.plot(r_pos/ARCSEC_TO_RAD, sample_mean, lw=1, color="r")
            plt.xlabel("r distance (arcsec)", fontsize = 20)
            plt.ylabel("Flux", fontsize = 20)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, header_name_for_file +"r_dist_flux.png"), dpi=200)
            plt.close()

            ## vis_real
            theta_test = sample_used[:,sample_list[0]]
            print(sample_for_plot(theta_test))
            q_dist_forp, d_real_mod_forp, d_imag_mod_forp, vis_model_forp, vis_model_imag_forp = gaussian_model_comparison(u_grid_1d, v_grid_1d, d_data, theta_test)
            plt.scatter(q_dist_forp, d_real_mod_forp, color="k", s =10)
            plt.errorbar(q_dist_forp, d_real_mod_forp, yerr = sigma_mat_1d_for_plot_real,  ls = "None",  color="k" )
            u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist_forp, d_real_mod_forp, 100)
            plt.scatter(u_bin, vis_bin,color="b", s=10)
            plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")

            for i in range(sample_num):
                theta_now = sample_used[:,sample_list[i]]
                q_pos, vis_sample = sample_gaussian_for_qspace(theta_now, np.max(q_dist_forp))
                plt.plot(q_pos, vis_sample, lw=1, color="r")

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, header_name_for_file +"real_comp.png"), dpi=200)
            plt.close()


            ## vis_imag
            plt.scatter(q_dist_forp, d_imag_mod_forp, color="k", s = 10)
            plt.errorbar(q_dist_forp, d_imag_mod_forp, yerr =  sigma_mat_1d_for_plot_imag, ls = "None",  color="k")
            u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist_forp, d_imag_mod_forp, 100)
            plt.scatter(u_bin, vis_bin,color="b", s=10)
            plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, header_name_for_file +"imag_comp.png"), dpi=200)
            plt.close()


def main_gauss_no_binning(DO_MCMC=True, TAKE_RANDOM_SAMPLE = True, MCMC_RUN=10000, NWALKER=32, \
    DO_SAMPLING=True, SAMPLE_NUM=50, NUM_BIN_FOR_DATA_LINEAR=200, NUM_BIN_FOR_DATA_LOG=200, MIN_BIN = 5,  \
    BIN_DATA_REPLACE =True, vis_folder = "./vis_data", outdir ="./out_dir_gauss", 
    target_object = None, NUMBER_OF_TARGETS = 1, N_process = 4):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count_ana_num = 0
    files = glob.glob(vis_folder + "/*")
    for file in files:

        if NUMBER_OF_TARGETS <= count_ana_num:
            break

        if target_object is not None and target_object != "all":


            if target_object not in file and count_ana_num ==0:
                count_ana_num = 0
                continue
            else:
                pass
        count_ana_num += 1
        header_name_for_file = file.split("/")[-1].replace(".npz", "")
        u_d, v_d, vis_d = load_obsdata(file)
        q_min_temp = np.min((u_d**2 + v_d**2)**0.5)
        q_max_temp = np.max((u_d**2 + v_d**2)**0.5)
        u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d, count_grid_1d = data_gridding.grid_2dvis_log10_edges_combination_of_two_gridding(vis_d, u_d, v_d, NUM_BIN_FOR_DATA_LINEAR, NUM_BIN_FOR_DATA_LOG, q_min_temp, q_max_temp, MIN_BIN)
        noise_grid_1d = np.median(noise_grid_1d * np.sqrt(count_grid_1d)) * np.ones(len(vis_d))

        sigma_mat_1d = 1/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
        sigma_mat_1d_for_plot_real = 1/sigma_mat_1d[:int(len(sigma_mat_1d)/2)]**0.5
        sigma_mat_1d_for_plot_imag = 1/sigma_mat_1d[int(len(sigma_mat_1d)/2):]**0.5
        d_data = np.append(vis_d.real, vis_d.imag)
        print(len(d_data))

        sample_out_name = os.path.join(outdir, header_name_for_file + "chain_mfreq")
        sample_out_name_npz = sample_out_name + ".npz"

        if DO_MCMC:
            delta_pre_fitting = determine_positional_offsets(d_data,  sigma_mat_1d, u_d, v_d, n_try = 500)
            initial_for_mcmc = make_initial_gauss(0.001, 1, 0.01*ARCSEC_TO_RAD, 1 * ARCSEC_TO_RAD, \
                delta_pre_fitting[0], delta_pre_fitting[1], NWALKER)
            n_w, n_para = np.shape(initial_for_mcmc)
            dtype = [("log_prior", float), ("log_likelihood", float), ("chi_sq", float)]
            with Pool(processes=N_process) as pool:
                sampler = emcee.EnsembleSampler(NWALKER, n_para, log_probability_gaussian_for_emcee, args=(u_d, v_d, \
                d_data, sigma_mat_1d), blobs_dtype=dtype,  pool=pool)
                sampler.run_mcmc(initial_for_mcmc, MCMC_RUN, progress=True)
            samples = sampler.get_chain(flat=True)
            blobs = sampler.get_blobs()
 
            np.savez(sample_out_name, sample = samples, log_prior =blobs["log_prior"], log_likelihood = blobs["log_likelihood"],\
             chi_sq = blobs["chi_sq"])

        if DO_SAMPLING:
            result_mcmc = np.load(sample_out_name_npz)
            sample = result_mcmc["sample"]
            chi_sq = np.ravel(result_mcmc["chi_sq"])
            lp = result_mcmc["log_prior"]# ("log_likelihood
            likeli = result_mcmc["log_likelihood"]
            post = np.ravel(lp) + np.ravel(likeli)
            mask = (sample[:,1]>0. * ARCSEC_TO_RAD) * (chi_sq>0) #* ((post>np.max(post)-20))
            sample = sample[mask]
            n_chain = len(sample)
            burn_out = n_chain - 10000
            plot_out_name_npz = os.path.join(outdir, header_name_for_file +"_mcmc.pdf")
            sample_for_plot_ = samples_for_plot_gaussian(sample[burn_out:])
            corner.corner(sample_for_plot_)
            plt.tight_layout()
            plt.savefig(plot_out_name_npz, dpi=200)
            plt.close()

            sample = np.load(sample_out_name_npz)["sample"][mask].T
            sample_used = sample[:,burn_out:]
            nx, nd = np.shape(sample_used)
            index_arr = np.arange(nd)
            sample_num = SAMPLE_NUM
            sample_list = np.array(random.sample(list(index_arr), sample_num))
            sample_mean_arr = []
            spec_arr = []

            ## flux
            for i in range(sample_num):
                theta_now = sample_used[:,sample_list[i]]
                r_pos, flux_one, flux_arcsec2 = sample_gaussian(theta_now)
                plt.plot(r_pos/ARCSEC_TO_RAD, flux_arcsec2, lw=0.1, color="k")
                sample_mean_arr.append(flux_arcsec2)
            sample_mean_arr = np.array(sample_mean_arr)
            sample_mean = np.mean(sample_mean_arr, axis =0)
            plt.plot(r_pos/ARCSEC_TO_RAD, sample_mean, lw=1, color="r")
            plt.xlabel("r distance (arcsec)", fontsize = 20)
            plt.ylabel("Flux", fontsize = 20)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, header_name_for_file +"r_dist_flux.png"), dpi=200)
            plt.close()

            ## vis_real
            theta_test = sample_used[:,sample_list[0]]
            print(sample_for_plot(theta_test))
            q_dist_forp, d_real_mod_forp, d_imag_mod_forp, vis_model_forp, vis_model_imag_forp = gaussian_model_comparison(u_d, v_d, d_data, theta_test)
            plt.scatter(q_dist_forp, d_real_mod_forp, color="k", s =10)
            plt.errorbar(q_dist_forp, d_real_mod_forp, yerr = sigma_mat_1d_for_plot_real,  ls = "None",  color="k" )
            u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist_forp, d_real_mod_forp, 100)
            plt.scatter(u_bin, vis_bin,color="b", s=10)
            plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")

            for i in range(sample_num):
                theta_now = sample_used[:,sample_list[i]]
                q_pos, vis_sample = sample_gaussian_for_qspace(theta_now, np.max(q_dist_forp))
                plt.plot(q_pos, vis_sample, lw=1, color="r")

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, header_name_for_file +"real_comp.png"), dpi=200)
            plt.close()


            ## vis_imag
            plt.scatter(q_dist_forp, d_imag_mod_forp, color="k", s = 10)
            plt.errorbar(q_dist_forp, d_imag_mod_forp, yerr =  sigma_mat_1d_for_plot_imag, ls = "None",  color="k")
            u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist_forp, d_imag_mod_forp, 100)
            plt.scatter(u_bin, vis_bin,color="b", s=10)
            plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, header_name_for_file +"imag_comp.png"), dpi=200)
            plt.close()