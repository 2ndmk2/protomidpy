from protomidpy import sample
from protomidpy import hankel
from protomidpy import mcmc_utils
from protomidpy import data_gridding
import matplotlib.pyplot as plt
import os
import corner
import numpy as np
ARCSEC_TO_RAD= 1/206265.0

def samples_for_plot(sample):
    sample_update = np.copy(sample)
    sample_update[0] = sample[0]/ARCSEC_TO_RAD
    sample_update[3] = sample[3] * 180/np.pi
    sample_update[1] = np.log10(sample[1])
    sample_update[2] = sample[2] 
    sample_update[4] = sample[4]/ARCSEC_TO_RAD
    sample_update[5] = sample[5]/ARCSEC_TO_RAD
    return sample_update

def mcmc_plot(sample_out_name_npz, outdir, header_name_for_file,n_sample = 5000):
    
    result_mcmc = np.load(sample_out_name_npz)
    sample = result_mcmc["sample"]
    log_likelihood= np.concatenate(result_mcmc["log_likelihood"])
    log_prior= np.concatenate(result_mcmc["log_prior"])
    likeli = np.ravel(result_mcmc["log_likelihood"])
    pos = log_likelihood+ log_prior
    pos_arg = np.argsort(pos)
    sample= sample[pos_arg[len(pos_arg) - n_sample:]]
    print(np.shape(sample))
    n_chain = len(sample)
    plot_out_name_npz = os.path.join(outdir, header_name_for_file +".pdf")
    corner.corner(sample)
    plt.tight_layout()
    plt.savefig(plot_out_name_npz, dpi=200)
    plt.close()


def sample_and_gallary(out_dir, log10_alpha, gamma, r_n, other_thetas, u_grid_1d, v_grid_1d, R_out, nrad, dpix,\
 d_data, sigma_mat_1d,  q_dist_model, H_mat_model, cov,  H_mat, V_A_minus1_U ,V_A_minus1_d, if_plot = False):

    r_dist = hankel.make_2d_mat_for_dist(r_n)
    if len(gamma)==1:
        theta_test = np.append([gamma[0], log10_alpha], other_thetas)
    if len(gamma )==2:
        theta_test = np.append([gamma[0], log10_alpha], other_thetas)  
        theta_test = np.append(theta_test, [gamma[1]])  
    sample_one = sample.sample_radial_profile_fixed_geo(cov, theta_test,r_dist, q_dist_model, H_mat_model, V_A_minus1_U ,V_A_minus1_d, H_mat)
    H_mat, q_dist, d_real_mod, d_imag_mod, vis_model_real, vis_model_imag, u_mod, v_mod= mcmc_utils.obs_model_comparison(sample_one, u_grid_1d, v_grid_1d, theta_test, d_data, R_out, nrad, dpix)
    q_model_extend = 10**(np.arange(4, 10, 0.01))
    model_vis_extend  = hankel.model_for_vis(sample_one, q_model_extend  , R_out, nrad, dpix, theta_test[2])
    u_bin, vis_bin, mean_err_bin, err_bin = data_gridding.bin_radial(q_dist, d_real_mod, 300)

    arg_sort = np.argsort(q_dist)
    if if_plot:
        fig = plt.figure()
        plt.subplot(4, 1,  1)
        plt.plot(r_n/ARCSEC_TO_RAD, sample_one)

        plt.subplot(4, 1,  2)
        plt.scatter(u_bin, vis_bin,color="b", s=10)
        plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
        plt.plot(q_model_extend, model_vis_extend ,zorder=100)
        plt.xlim(10**6, 10**8)
        plt.ylim(-0.006, 0.006)
        plt.xscale("log")
        plt.tight_layout()

        plt.subplot(4, 1,  3)
        plt.scatter(u_bin, vis_bin,color="b", s=10)
        plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
        plt.plot(q_model_extend, model_vis_extend ,zorder=100)
        plt.xlim(10**6, 10**8)
        plt.ylim(-0.02, 0.02)
        plt.xscale("log")
        plt.tight_layout()

        plt.subplot(4, 1,  4)
        plt.plot(q_dist[arg_sort], vis_model_real[arg_sort], lw=1, color="r")
        plt.scatter(u_bin, vis_bin,color="b", s=10)
        plt.errorbar(u_bin, vis_bin, yerr =  mean_err_bin, ls = "None",  color="b")
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "alpha%.2f_gamma%.2f.pdf" % (log10_alpha, gamma[0])))
        plt.close()
    np.savez(os.path.join(out_dir, "alpha%.2f_gamma%.2f" % (log10_alpha, gamma[0])), \
        r_n = r_n/ARCSEC_TO_RAD, flux_model = sample_one, q_dist = q_dist[arg_sort], vis_model = vis_model_real[arg_sort])

def plotter_evidence(evidence_mat, term1_mat,term2_mat, term3_mat, out_dir, head_name, percen_mat = [25,  100], percen_mat_term = [0,  75]):


    evidence_mat = np.array(evidence_mat)
    percen_mat_val = np.percentile(evidence_mat, percen_mat)
    percen_mat_mat1 = np.percentile(term1_mat,percen_mat)
    percen_mat_mat2 = np.percentile(term2_mat, percen_mat )
    percen_mat_mat3 = np.percentile(term3_mat, percen_mat)

    figure = plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(evidence_mat)
    plt.colorbar()
    plt.subplot(2,4,2)
    plt.imshow(evidence_mat, vmin=percen_mat_val[0], vmax=percen_mat_val[1])
    plt.colorbar()
    plt.subplot(2,4,3)
    plt.imshow(gamma_mat)    
    plt.colorbar()
    plt.subplot(2,4,4)
    plt.imshow(alpha_mat)    
    plt.colorbar()
    plt.subplot(2,4,5)
    plt.imshow(term1_mat, vmin=percen_mat_mat1[0], vmax=percen_mat_mat1[1])
    plt.colorbar()
    plt.subplot(2,4,6)
    plt.imshow(term2_mat, vmin=percen_mat_mat2[0], vmax=percen_mat_mat2[1])
    plt.colorbar()
    plt.subplot(2,4,7)
    plt.imshow(term3_mat, vmin=percen_mat_mat3[0], vmax=percen_mat_mat3[1])
    plt.colorbar()
    plt.savefig("./%s/%s_evidence.png" % (out_dir, head_name))
    plt.close()
    #plt.show()

