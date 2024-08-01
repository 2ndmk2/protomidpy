import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy import interpolate
import os
import matplotlib.patches as patches

ARCSEC_TO_RAD= 1/206265.0


def make_alpha_gamma_arr_for_plot(alpha_arr, gamma_arr, alpha_i, gamma_i, wd_side):
    alpha_select = np.flip(alpha_arr[alpha_i-wd_side :alpha_i+wd_side+1])
    gamma_select = np.flip(gamma_arr[gamma_i-wd_side :gamma_i+wd_side+1])
    return alpha_select, gamma_select 


def choose_best_from_grid(alpha_arr, gamma_arr, evidence_mat):
    gamma_i, alpha_i = np.unravel_index(np.argmax(evidence_mat), evidence_mat.shape)
    return alpha_i, gamma_i


def make_corner_values(arr):
    delta_value = arr[1] -arr[0]
    arr = np.append([arr[0]-delta_value*0.5],np.array(arr)+ delta_value*0.5)
    return arr


def conv_flux(flux_interp, sigma=1.7):
    conv_flux = gaussian_filter1d(flux_interp, sigma=sigma)
    return conv_flux


def load_input_model(outdir = "./simdata", name_model= "sim_1.50"):

    model_input = np.load(os.path.join( outdir, "%s_model.npz" % name_model))
    r_pos_input = model_input["r_pos"]
    flux_input = model_input["flux"]
    r_pos_input_mod = np.append([0], r_pos_input)
    flux_input_mod = np.append([flux_input[0]], flux_input)
    r_pos_input_mod = np.append(r_pos_input_mod, [10])
    flux_input_mod = np.append(flux_input_mod , [0])

    return r_pos_input_mod, flux_input_mod

def interp_for_flux(r_pos, flux, rout = 2.5, N = 150):
    f_int_func  = interpolate.interp1d(r_pos, flux)
    r_interp= np.linspace(0, rout,N)
    flux_interp = f_int_func (r_interp)
    return r_interp, flux_interp,f_int_func 


def plot_evidence(alpha_arr, gamma_arr, evidence_mat, out_dir, head_name , alpha_i=2, gamma_i=2,  wd_side = 1, title =""):

    d_alpha = np.abs(alpha_arr[1] -  alpha_arr[0])
    d_gamma = np.abs(gamma_arr[1] -  gamma_arr[0])
    alpha_lw = alpha_arr[alpha_i] - (0.5 + wd_side ) * d_alpha
    gamma_lw = gamma_arr[gamma_i] - (0.5 + wd_side ) * d_gamma
    alpha_wd = (2 * wd_side + 1) * d_alpha 
    gamma_wd = (2 * wd_side + 1) * d_gamma
    fig, ax = plt.subplots(figsize = (7,6))
    ARCSEC_TO_RAD= 1/206265.0
    plt.rcParams.update({'font.size': 14})
    aximg = plt.pcolor(make_corner_values(alpha_arr), make_corner_values(gamma_arr), evidence_mat -np.max( evidence_mat ) , vmin =-500, vmax=0)
    plt.xlabel("$\log_{10} \\alpha$", fontsize = 18)
    plt.ylabel("$\\gamma\; (arcsec)$", fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title(title, fontsize = 18)
    cbar = fig.colorbar(aximg , ax=ax)
    cbar.set_label("$\log[p(d|\\theta, g)]- {\\rm const}$", size=20)
    ax.add_patch(
         patches.Rectangle(
            (alpha_lw, gamma_lw),
            alpha_wd ,
            gamma_wd,
            edgecolor = 'r',
            facecolor = None,
            fill=False,
            lw=3
         ) )
    plt.savefig( os.path.join(out_dir, "%s_evidence.pdf" % head_name ))
    plt.show()

def plot_input_out(file_flux, r_interp, flux_interp, f_int_func , out_dir, head_name, figsize = (12, 8), fontsize_plot  =20, fontsize_ticks = 20,  xlims = [0,2.5]):

    fig1 = plt.figure(figsize=figsize)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    plt.plot(file_flux["r_n"], file_flux["flux_model"], color="tab:blue", alpha = 0.5)
    plt.plot(r_interp, flux_interp, color="tab:orange", label="input profile")
    plt.xlim(xlims[0], xlims[1])
    plt.ylabel(r"Intensity [1/arcsec$^2$]",fontsize = fontsize_plot )
    plt.yticks(fontsize = fontsize_ticks )
    plt.xticks(color="None" )
    plt.legend(fontsize = 20)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(file_flux["r_n"], f_int_func (file_flux["r_n"]) - file_flux["flux_model"], color="k", alpha = 0.5)
    plt.xlim(xlims[0], xlims[1])
    plt.yticks(fontsize = fontsize_ticks )
    plt.xticks(fontsize = fontsize_ticks)
    plt.xlabel(r"Radial distance (arcsec)",fontsize = fontsize_plot )
    plt.ylabel(r"Residual",fontsize = fontsize_plot )
    plt.savefig( os.path.join(out_dir, "%s_plot_radial.pdf" % head_name ), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_2d_flux_arrays(alpha_select, gamma_select, load_dir, out_dir, head_name, f_int_func_input):
    fig, axs = plt.subplots(len(alpha_select),len(gamma_select), figsize=(len(alpha_select)*5, len(gamma_select)*3),  constrained_layout=True,
                        sharex=True, sharey=True)
    for i_args_div in range(len(gamma_select)):
        for i_args_mod in range(len(alpha_select)):
            data = np.load(os.path.join(load_dir,"alpha%.2f_gamma%.2f.npz" % (alpha_select[i_args_mod], gamma_select[i_args_div])))
            r_n = data["r_n"]
            flux_model = data["flux_model"]
            flux_interp =f_int_func_input(r_n)
            ax1 = axs[i_args_div][i_args_mod]
            ax1.set_title('$(\log_{10}\\alpha, \\gamma) =( %d, %.2f$") ' % (alpha_select[i_args_mod], gamma_select[i_args_div]))
            ax1.xaxis.set_tick_params(labelsize = 15)
            ax1.yaxis.set_tick_params(labelsize = 15)
            ax1.plot(r_n, (flux_interp-  flux_model),  lw=3, color="tab:blue", label="$r \\times (I_{\\rm true} - I_{\\rm model})$",zorder =100)
                
    fig.supxlabel('radial distance (arcsec)', fontsize = 25)
    fig.supylabel('$ (I_{\\rm true} - I_{\\rm recovered})$', fontsize = 25)
    plt.savefig(os.path.join(out_dir, "%s_radial_profile_sim.pdf" % head_name), dpi = 100, bbox_inches='tight')
    plt.show()


def plot_2d_flux_for_data_arrays(alpha_select, gamma_select, load_dir, out_dir, head_name):
    fig, axs = plt.subplots(len(alpha_select),len(gamma_select), figsize=(len(alpha_select)*5, len(gamma_select)*3),  constrained_layout=True,
                        sharex=True, sharey=True)
    for i_args_div in range(len(gamma_select)):
        for i_args_mod in range(len(alpha_select)):
            data = np.load(os.path.join(load_dir,"alpha%.2f_gamma%.2f.npz" % (alpha_select[i_args_mod], gamma_select[i_args_div])))
            r_n = data["r_n"]
            flux_model = data["flux_model"]
            ax1 = axs[i_args_div][i_args_mod]
            ax1.set_title('$(\log_{10}\\alpha, \\gamma) =( %d, %.2f$") ' % (alpha_select[i_args_mod], gamma_select[i_args_div]))
            ax1.xaxis.set_tick_params(labelsize = 15)
            ax1.yaxis.set_tick_params(labelsize = 15)
            ax1.plot(r_n,  flux_model,  lw=3, color="tab:blue", label="$r \\times (I_{\\rm true} - I_{\\rm model})$",zorder =100)
            plt.ylim(-0.01, 0.01)
    fig.supxlabel('radial distance (arcsec)', fontsize = 25)
    fig.supylabel('$ (I_{\\rm true} - I_{\\rm recovered})$', fontsize = 25)
    plt.savefig(os.path.join(out_dir, "%s_radial_profile_sim.pdf" % head_name), dpi = 100, bbox_inches='tight')
    plt.show()



def plot_2d_fluxrad_arrays(alpha_select, gamma_select, load_dir, out_dir, head_name, f_int_func_input):
    fig, axs = plt.subplots(len(alpha_select),len(gamma_select), figsize=(len(alpha_select)*5, len(gamma_select)*3),  constrained_layout=True,
                        sharex=True, sharey=True)
    for i_args_div in range(len(gamma_select)):
        for i_args_mod in range(len(alpha_select)):
            data = np.load(os.path.join(load_dir,"alpha%.2f_gamma%.2f.npz" % (alpha_select[i_args_mod], gamma_select[i_args_div])))
            r_n = data["r_n"]
            flux_model = data["flux_model"]
            flux_interp = f_int_func_input(r_n)
            ax1 = axs[i_args_div][i_args_mod]
            ax1.set_ylim(-0.015, 0.015)
            ax1.set_title('$(\log_{10}\\alpha, \\gamma) =( %d, %.2f$") ' % (alpha_select[i_args_mod], gamma_select[i_args_div]))
            ax1.xaxis.set_tick_params(labelsize = 15)
            ax1.yaxis.set_tick_params(labelsize = 15)
            ax1.plot(r_n, (flux_interp-  flux_model)*r_n,  lw=3, color="tab:blue", label="$r \\times (I_{\\rm true} - I_{\\rm model})$",zorder =100)
                
    fig.supxlabel('radial distance (arcsec)', fontsize = 25)
    fig.supylabel('$ r\\times (I_{\\rm true} - I_{\\rm recovered})$', fontsize = 25)
    plt.savefig(os.path.join(out_dir, "%s_r_flux_profile_sim.pdf" % head_name), dpi = 100, bbox_inches='tight')
    plt.show()


def plot_2d_vis_arrays(alpha_select, gamma_select, u_bin, vis_bin, mean_err_bin, load_dir, out_dir, head_name):
    fig, axs = plt.subplots(len(alpha_select),len(gamma_select), figsize=(len(alpha_select)*5, len(gamma_select)*4),  constrained_layout=True,
                            sharex=True, sharey=True)

    for i_args_mod in range(len(alpha_select)):
        for i_args_div in range(len(gamma_select)):
            data = np.load(os.path.join(load_dir,"alpha%.2f_gamma%.2f.npz" % (alpha_select[i_args_mod], gamma_select[i_args_div])))
            q_dist = data["q_dist"]
            vis_model = data["vis_model"]
            axs[i_args_div][i_args_mod].plot(q_dist/1e6, vis_model*1000, color="tab:blue", zorder = 100, alpha = 0.7, lw = 2, label="model")
            axs[i_args_div][i_args_mod].errorbar(u_bin/1e6, vis_bin*1000, yerr = mean_err_bin*1000, fmt="o", color="tab:orange", label="data")
            axs[i_args_div][i_args_mod].set_ylim(-4, 4)
            axs[i_args_div][i_args_mod].set_title('$(\log_{10}\\alpha, \\gamma) =( %d, %.2f$") ' % (alpha_select[i_args_mod], gamma_select[i_args_div]))
            axs[i_args_div][i_args_mod].set_xlim(1, 10)
            axs[i_args_div][i_args_mod].xaxis.set_tick_params(labelsize = 20)
            axs[i_args_div][i_args_mod].yaxis.set_tick_params(labelsize = 20)
            if i_args_div ==0 and i_args_mod ==0:
                axs[i_args_div][i_args_mod].legend(fontsize = 16)   
            plt.xscale("log")

    fig.supxlabel('uv distance $(M\\lambda)$', fontsize = 30)
    fig.supylabel('Real part of Visibility (mJy)', fontsize = 30)
    plt.savefig(os.path.join(out_dir, "%s_q_vis_sim.pdf" % head_name), dpi = 100, bbox_inches='tight')
    plt.show()
