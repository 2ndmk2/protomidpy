import numpy as np
import os
from georadial import hankel
from scipy import interpolate


ARCSEC_TO_RAD= 1/206265.0

def load_visdata(filename):
    data_raw = np.load(filename)        
    u_d = data_raw["u_obs"]
    v_d = data_raw["v_obs"]
    vis = data_raw["vis_obs"]
    wgt = data_raw["wgt_obs"]
    sigma = 1/wgt**0.5
    return u_d, v_d, vis, wgt, sigma

def calc_beamarea(maj, minor):
    maj_sigma = maj/(2 * np.sqrt(2 * np.log(2)))
    minor_sigma = minor/(2 * np.sqrt(2 * np.log(2)))
    beam_area = np.pi/(4*np.log(2))*maj*minor

    return beam_area

def load_profile(filename, beam_maj, beam_min, r_max =100):

    #model load
    file = open(filename)
    lines = file.readlines()
    beam_area = calc_beamarea(beam_maj, beam_min)
    r_arcsec_arr = []
    flux_arr = []
    for line in lines:
        itemList = line.split()
        r_arcsec_arr.append(float(itemList[1])*ARCSEC_TO_RAD)
        flux_arr.append(float(itemList[2])/beam_area)
    r_arcsec_arr = np.append(0, r_arcsec_arr)
    flux_arr = np.append(flux_arr[0], flux_arr)
    r_arcsec_arr = np.append(r_arcsec_arr, r_max * ARCSEC_TO_RAD)
    flux_arr = np.append(flux_arr, 0)
    flux_fanc = interpolate.interp1d(r_arcsec_arr, flux_arr)
    return r_arcsec_arr, flux_arr, flux_fanc

def model_extend(flux_fanc, a):
    flux_mod = lambda x:  flux_fanc(a*x)
    return flux_mod

def resampling_model(flux_fanc, R_out, N):
    r_n, jn, qmax, q_n = hankel.make_collocation_points(R_out, N)
    flux_rn = flux_fanc(r_n)
    return r_n, flux_rn
    
def transform_for_computing_vis(u_d, v_d, pa, cosi, R_out, N):
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d = -cos_pa * u_d + sin_pa *v_d
    v_new_d = -sin_pa * u_d - cos_pa *v_d
    u_new_d = u_new_d * cosi
    q_dist = (u_new_d**2 + v_new_d **2)**0.5
    H_mat = hankel.make_hankel_matrix(q_dist,  R_out, N, cosi)
    return q_dist, H_mat

def add_noise_to_data(vis_model, sigma):
    vis_real_model_noise_added = vis_model + sigma * np.random.randn(len(vis_model))
    vis_imag_model_noise_added = sigma * np.random.randn(len(vis_model))
    vis_complex = vis_real_model_noise_added + 1j * vis_imag_model_noise_added
    return vis_real_model_noise_added, vis_imag_model_noise_added, vis_complex

def save_sim(file_head, r_n, flux_r_n, flux_raw, u_obs, v_obs, vis_obs, wgt_obs):
    np.savez(file_head + "_vis", u_obs = u_obs, v_obs = v_obs,vis_obs = vis_obs, wgt_obs  = wgt_obs )
    np.savez(file_head + "_model", r_pos = r_n, flux =flux_r_n, flux_raw = flux_raw)
