import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import os
from scipy import optimize

from georadial.covariance import ARCSEC_TO_RAD



def log_gridding_1d(xmin, xmax, n_d_log):

    dx = (np.log10(xmax) - np.log10(xmin))/(n_d_log+1.0)
    xedges = np.linspace(np.log10(xmin), np.log10(xmax), n_d_log+1)
    xedges_for_meshgrid_side = 0.5*dx  + np.linspace(np.log10(xmin), np.log10(xmax), n_d_log+1)
    xedges_for_meshgrid_side = xedges_for_meshgrid_side[:n_d_log]    
    coord_for_grid = 10**xedges
    coord_for_grid = np.append(coord_for_grid, 0)
    rep_positions_for_grid = 10**xedges_for_meshgrid_side 
    rep_positions_for_grid = np.append(rep_positions_for_grid,  0.5 * xmin)
    coord_for_grid = np.sort(coord_for_grid)
    rep_positions_for_grid = np.sort(rep_positions_for_grid)
    return coord_for_grid

def linear_gridding_1d(xmin, xmax, n_d_log):

    dx = (xmax - xmin)/(n_d_log+1.0)
    xedges = np.linspace(xmin, xmax, n_d_log+1)
    xedges_for_meshgrid_side = 0.5*dx  + np.linspace(xmin, xmax, n_d_log+1)
    xedges_for_meshgrid_side = xedges_for_meshgrid_side[:n_d_log]    
    coord_for_grid = xedges
    coord_for_grid = np.append(coord_for_grid, 0)
    rep_positions_for_grid = xedges_for_meshgrid_side 
    rep_positions_for_grid = np.append(rep_positions_for_grid,  0.5 * xmin)
    coord_for_grid = np.sort(coord_for_grid)
    rep_positions_for_grid = np.sort(rep_positions_for_grid)
    return coord_for_grid


def data_binning_1d(x, y, weights, grid):
    weight_sum, xedges, binnumber =  binned_statistic( x, weights, 'sum', bins=grid)
    vis_real_sum, xedges, binnumber =  binned_statistic(x, y.real * weights, 'sum', bins=grid)
    vis_imag_sum, xedges, binnumber =  binned_statistic(x, y.imag * weights,  'sum', bins=grid)
    x_sum, xedges, binnumber =  binned_statistic(x, x * weights,  'sum', bins=grid)

    mask = weight_sum>0
    vis_real_masked = np.ravel(vis_real_sum[mask])/np.ravel(weight_sum[mask])    
    vis_imag_masked = np.ravel(vis_imag_sum[mask])/np.ravel(weight_sum[mask])
    x_masked = np.ravel(x_sum[mask])/np.ravel(weight_sum[mask])
    vis_grid_1d = vis_real_masked  + 1j * vis_imag_masked 
    weight_grid_1d = np.ravel(weight_sum[mask])
    noise_grid_1d = (1/weight_grid_1d**0.5) + 1j * (1/weight_grid_1d **0.5)
    sigma_mat_1d = 1.0/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
    d_data = np.append(vis_grid_1d.real, vis_grid_1d.imag)
    return x_masked, vis_grid_1d, noise_grid_1d, d_data, sigma_mat_1d


def take_rep(x):
    x_rep = []
    for i in range(len(x)-1):
        x_rep.append( (x[i] + x[i+1] )/2.0)
    return np.array(x_rep)


def log_gridding_2d(xmin,xmax,  n_d_log):

    dx = (np.log10(xmax) - np.log10(xmin))/(n_d_log+1.0)
    xedges = np.linspace(np.log10(xmin), np.log10(xmax), n_d_log+1)   
    xedges_for_meshgrid_side = 0.5*dx  + np.linspace(np.log10(xmin), np.log10(xmax), n_d_log+1)
    xedges_for_meshgrid_side = xedges_for_meshgrid_side[:n_d_log]
    coord_for_grid = np.append(10**xedges, -10 ** xedges)
    coord_for_grid = np.append(coord_for_grid, 0)
    coord_for_grid = np.sort(coord_for_grid)
    rep_positions_for_grid = take_rep(coord_for_grid)
    uu_for_grid_pos, vv_for_grid_pos= np.meshgrid(rep_positions_for_grid, rep_positions_for_grid)
    return coord_for_grid, rep_positions_for_grid, uu_for_grid_pos, vv_for_grid_pos

def linear_gridding_2d(xmin,xmax,  n_d_log):

    dx = (xmax - xmin)/(n_d_log+1.0)
    xedges = np.linspace(xmin, xmax, n_d_log+1)
    xedges_for_meshgrid_side = 0.5*dx  + np.linspace(xmin, xmax, n_d_log+1)
    xedges_for_meshgrid_side = xedges_for_meshgrid_side[:n_d_log]  
    coord_for_grid = xedges
    coord_for_grid = np.append(xedges, - xedges)
    coord_for_grid = np.append(coord_for_grid, 0)
    coord_for_grid = np.sort(coord_for_grid)
    rep_positions_for_grid = np.append(xedges_for_meshgrid_side, - xedges_for_meshgrid_side)
    rep_positions_for_grid = np.append(rep_positions_for_grid, [-0.5 * xmin, 0.5 * xmin])
    rep_positions_for_grid = np.sort(rep_positions_for_grid)
    uu_for_grid_pos, vv_for_grid_pos= np.meshgrid(rep_positions_for_grid, rep_positions_for_grid)
    return coord_for_grid, rep_positions_for_grid, uu_for_grid_pos, vv_for_grid_pos


def data_binning_2d(x1, x2, y, weights, grid):

    weight_sum, xedges, yedges, binnumber = binned_statistic_2d(x1, x2,weights, statistic="sum", bins=[grid, grid])      
    vis_real_sum, xedges, yedges, binnumber = binned_statistic_2d(x1, x2, y.real * weights, statistic="sum", bins=[grid, grid])      
    vis_imag_sum, xedges, yedges, binnumber = binned_statistic_2d(x1, x2, y.imag * weights, statistic="sum",  bins=[grid, grid]) 
    count_sum, xedges, yedges, binnumber = binned_statistic_2d(x1, x2,weights, statistic="count", bins=[grid, grid])      
    x1_sum, xedges, yedges, binnumber = binned_statistic_2d(x1, x2, x1 * weights, statistic="sum", bins=[grid, grid])      
    x2_sum, xedges, yedges, binnumber = binned_statistic_2d(x1, x2, x2 * weights, statistic="sum", bins=[grid, grid])      


    mask = weight_sum>0
    vis_real_masked = np.ravel(vis_real_sum[mask])/np.ravel(weight_sum[mask])
    vis_imag_masked = np.ravel(vis_imag_sum[mask])/np.ravel(weight_sum[mask])
    x1_masked = np.ravel(x1_sum[mask])/np.ravel(weight_sum[mask])
    x2_masked = np.ravel(x2_sum[mask])/np.ravel(weight_sum[mask])
    vis_grid_1d = vis_real_masked  + 1j * vis_imag_masked 
    weight_grid_1d = np.ravel(weight_sum[mask])
    count_sum = count_sum [mask]
    noise_grid_1d = (1/weight_grid_1d**0.5) + 1j * (1/weight_grid_1d **0.5)
    sigma_mat_1d = 1.0/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
    d_data = np.append(vis_grid_1d.real, vis_grid_1d.imag)

    return x1_masked, x2_masked, vis_grid_1d, noise_grid_1d, sigma_mat_1d, d_data, count_sum



def deproject_radial(u_d, v_d, vis_d, wgt_d, cosi, pa, delta_x, delta_y):

    delta_x_rad  = delta_x * ARCSEC_TO_RAD
    delta_y_rad  = delta_y * ARCSEC_TO_RAD
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    q = ((cosi **2) * (-u_d* cos_pa + v_d * sin_pa)**2+ (- u_d * sin_pa - v_d *cos_pa)**2)**0.5
    vis_d = vis_d/cosi
    wgt_d =  wgt_d/(cosi**2)
    sigma_d = np.append(wgt_d, wgt_d)
    noise_d = 1/np.sqrt(wgt_d) + 1j * 1/np.sqrt(wgt_d) 
    q_v_zero = np.zeros(len(q))

    ###
    diag_mat_cos_inv = np.cos(2 * np.pi * (delta_x_rad * u_d + delta_y_rad * v_d))
    diag_mat_sin_inv = np.sin(2 * np.pi * (delta_x_rad * u_d + delta_y_rad * v_d))
    d_real = vis_d.real
    d_imag = vis_d.imag
    d_real_mod = d_real * diag_mat_cos_inv  - d_imag * diag_mat_sin_inv
    d_imag_mod = +  d_real * diag_mat_sin_inv + d_imag * diag_mat_cos_inv
    d_data = np.append(d_real_mod, d_imag_mod)
    vis_d_new =d_real_mod  + 1j * d_imag_mod

    return q, q_v_zero ,vis_d_new , wgt_d, noise_d, sigma_d, d_data

def deproject_radial_2d(u_d, v_d, vis_d, wgt_d, cosi, pa, delta_x, delta_y):

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    vis_real = vis_d.real
    vis_imag = vis_d.imag
    u_new = cosi * (-u_d* cos_pa + v_d * sin_pa)
    v_new = - u_d * sin_pa - v_d *cos_pa
    vis_d = vis_d/cosi
    wgt_d =  wgt_d/(cosi**2)
    sigma_d = np.append(wgt_d, wgt_d)
    noise_d = 1/np.sqrt(wgt_d) + 1j * 1/np.sqrt(wgt_d) 
    diag_mat_cos_inv = np.cos(2 * np.pi * (delta_x * u_d + delta_y * v_d))
    diag_mat_sin_inv = np.sin(2 * np.pi * (delta_x * u_d + delta_y * v_d))
    d_real = vis_d.real
    d_imag = vis_d.imag
    d_real_mod = d_real * diag_mat_cos_inv  + d_imag * diag_mat_sin_inv
    d_imag_mod = +  d_real * diag_mat_sin_inv - d_imag * diag_mat_cos_inv
    d_data = np.append(d_real_mod, d_imag_mod)

    return u_new, v_new, vis_d, wgt_d, noise_d, sigma_d, d_data


def deproject_radial_and_bin(gridfile, u_d, v_d, vis_d, wgt_d, cosi, pa, delta_x, delta_y, n_bin_linear=400, n_bin_log=400, q_min_max_bin = None):

    if q_min_max_bin is None:
        q_min_temp = np.min((u_d**2 + v_d**2)**0.5)
        q_max_temp = np.max((u_d**2 + v_d**2)**0.5)
    else:
        q_min_temp = q_min_max_bin[0]
        q_max_temp = q_min_max_bin[1]
        
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    vis_real = vis_d.real
    vis_imag = vis_d.imag
    q = ((cosi **2) * (-u_d* cos_pa + v_d * sin_pa)**2+ (- u_d * sin_pa - v_d *cos_pa)**2)**0.5
    vis_d = vis_d/cosi
    wgt_d =  wgt_d/(cosi**2)
    q_grid_1d, vis_grid_1d, noise_grid_1d = grid_data_lilearlog_for_1d(vis_d, q, wgt_d, n_bin_linear, n_bin_log, q_min_temp, q_max_temp)
    q_v_zero = np.zeros(len(q_grid_1d))
    sigma_mat_1d = 1.0/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
    d_data = np.append(vis_grid_1d.real, vis_grid_1d.imag)
    np.savez(gridfile.replace("npz",""), u =q_grid_1d, v = q_v_zero, vis = vis_grid_1d, noise = noise_grid_1d)
    return q_grid_1d, q_v_zero , vis_grid_1d, noise_grid_1d, sigma_mat_1d, d_data

def get_gridded_obs_data(gridfile, u_d, v_d, vis_d, wgt_d,n_bin_linear, n_bin_log, q_min_max_bin, replace =True):
    if q_min_max_bin is None:
        q_min_temp = np.min((u_d**2 + v_d**2)**0.5)
        q_max_temp = np.max((u_d**2 + v_d**2)**0.5)
    else:
        q_min_temp = q_min_max_bin[0]
        q_max_temp = q_min_max_bin[1]

    if os.path.exists(gridfile) and replace:
        grid_data = np.load(gridfile)
        u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = grid_data["u"], grid_data["v"], grid_data["vis"], grid_data["noise"]
        q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
        q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)            
    else:
        u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = grid_2dvis_log10_edges_combination_of_two_gridding_using_weights(vis_d, u_d, v_d, wgt_d, n_bin_linear, n_bin_log, q_min_temp, q_max_temp)
        np.savez(gridfile.replace("npz",""), u = u_grid_1d, v = v_grid_1d, vis = vis_grid_1d, noise = noise_grid_1d)
        q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
        q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)
        u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = grid_2dvis_log10_edges_combination_of_two_gridding_using_weights(vis_d, u_d, v_d, wgt_d, n_bin_linear, n_bin_log, q_min, q_max)

    sigma_mat_1d = 1.0/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
    d_data = np.append(vis_grid_1d.real, vis_grid_1d.imag)
    return u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d, sigma_mat_1d, d_data

def get_gridded_obs_data(gridfile, u_d, v_d, vis_d, wgt_d,n_bin_linear, n_bin_log, q_min_max_bin, replace =True):
    if q_min_max_bin is None:
        q_min_temp = np.min((u_d**2 + v_d**2)**0.5)
        q_max_temp = np.max((u_d**2 + v_d**2)**0.5)
    else:
        q_min_temp = q_min_max_bin[0]
        q_max_temp = q_min_max_bin[1]

    if os.path.exists(gridfile) and replace:
        grid_data = np.load(gridfile)
        u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = grid_data["u"], grid_data["v"], grid_data["vis"], grid_data["noise"]
        q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
        q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)            
    else:
        u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = grid_2dvis_log10_edges_combination_of_two_gridding_using_weights(vis_d, u_d, v_d, wgt_d, n_bin_linear, n_bin_log, q_min_temp, q_max_temp)
        np.savez(gridfile.replace("npz",""), u = u_grid_1d, v = v_grid_1d, vis = vis_grid_1d, noise = noise_grid_1d)
        q_min = np.min((u_grid_1d**2 + v_grid_1d**2)**0.5)
        q_max = np.max((u_grid_1d**2 + v_grid_1d**2)**0.5)
        u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d = grid_2dvis_log10_edges_combination_of_two_gridding_using_weights(vis_d, u_d, v_d, wgt_d, n_bin_linear, n_bin_log, q_min, q_max)

    sigma_mat_1d = 1.0/(np.append(noise_grid_1d.real, noise_grid_1d.imag)**2)
    d_data = np.append(vis_grid_1d.real, vis_grid_1d.imag)
    return u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d, sigma_mat_1d, d_data





def obs_q_and_vis(u_d, v_d,  d_data, cosi, pa, delta_x, delta_y):

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
    d_imag_mod = +  d_real * diag_mat_sin_inv + d_imag * diag_mat_cos_inv
    return q_dist , d_real_mod, d_imag_mod

def bin_obs_q_and_vis(u_d, v_d,  d_data, cosi, pa, delta_x, delta_y):
    q_dist, d_real_mod, d_imag_mod = obs_q_and_vis(u_d, v_d,  d_data, cosi, pa, delta_x, delta_y)
    u_bin, vis_bin, mean_err_bin, err_bin = bin_radial(q_dist, d_real_mod, 300)
    return u_bin, vis_bin, mean_err_bin, err_bin 



def bin_radial(u_dist, vis_real, N):
    """ Calculate spectral index between two frequencies

    Args: 
        u_dist (ndarray): 1d array for visibility distance
        vis_real (ndarray): 1d array for visibility data
        N (int): number of radial points
    Returns:
        u_bin (ndarray): 1d array for binned visibility distance
        vis_bin (ndarray): 1d array for binned visibility
        mean_err_bin (ndarray): 1d array for binned error 
        err_bin (ndarray): 1d array for std for data

    """
    qstart = 0
    qend = np.max(u_dist)
    u_arr = []
    vis_arr = []
    mean_sigma_vis_arr = []
    std_vis_arr = []
    d_q = (qend - qstart)/float(N)
    for i in range(N):
        q_1 = qstart + i *d_q
        q_2 = qstart + (i+1) *d_q
        flag = (u_dist > q_1) * (u_dist < q_2) * np.isfinite(vis_real)
        num_q = len(u_dist[flag])
        if num_q >2:
            u_arr.append(np.mean(u_dist[flag]))
            vis_arr.append(np.mean(vis_real[flag]))
            mean_sigma_vis_arr.append(np.std(vis_real[flag])/np.sqrt(num_q-1))
            std_vis_arr.append(np.std(vis_real[flag]))

    u_bin = np.array(u_arr)
    vis_bin = np.array(vis_arr)
    mean_err_bin = np.array(mean_sigma_vis_arr)
    err_bin = np.array(std_vis_arr)
    return u_bin, vis_bin, mean_err_bin, err_bin






####
def calc_boundary_bewteen_log_and_linear(xmax, xmin, n_d_eq, n_d_log):
    d_q = xmax/n_d_eq
    lambda_f = lambda x: (x/n_d_log) * np.log(x/xmin) - d_q
    sol = optimize.root_scalar(lambda_f, bracket=[1, 10**10], method='brentq')
    x_max_for_log_grid = sol.root
    d_q_log = np.log10(x_max_for_log_grid/xmin)/n_d_log
    d_q_log_max = x_max_for_log_grid * 10**( d_q_log) - x_max_for_log_grid
    return x_max_for_log_grid


def making_grids_nufft_combination_of_two_gridding_for_1d(xmax, xmin, n_d_eq, n_d_log):
    
    d_linear_x = xmax/n_d_eq
    x_max_for_log_grid = calc_boundary_bewteen_log_and_linear(xmax, xmin, n_d_eq, n_d_log)
    x_coord_linear = np.arange(x_max_for_log_grid, x_max_for_log_grid +  d_linear_x*n_d_eq, step = d_linear_x)
    x_coord_linear = x_coord_linear[x_coord_linear-d_linear_x<xmax]
    x_coord_linear = x_coord_linear[1:]
    xedges_for_meshgrid_side_linear = x_coord_linear - 0.5*d_linear_x 
        
    dx = (np.log10(x_max_for_log_grid) - np.log10(xmin))/(n_d_log+1.0)
    xedges = np.linspace(np.log10(xmin), np.log10(x_max_for_log_grid), n_d_log+1)
    xedges_for_meshgrid_side = 0.5*dx  + np.linspace(np.log10(xmin), np.log10(x_max_for_log_grid), n_d_log+1)
    xedges_for_meshgrid_side = xedges_for_meshgrid_side[:n_d_log]
    ##
    coord_for_grid = 10**xedges
    coord_for_grid = np.append(coord_for_grid, 0)
    coord_for_grid = np.append(coord_for_grid, x_coord_linear)
    coord_for_grid = np.sort(coord_for_grid)

    rep_positions_for_grid = 10**xedges_for_meshgrid_side 
    rep_positions_for_grid = np.append(rep_positions_for_grid,  0.5 * xmin)
    rep_positions_for_grid = np.append(rep_positions_for_grid, xedges_for_meshgrid_side_linear)
    rep_positions_for_grid = np.sort(rep_positions_for_grid)
    return coord_for_grid, rep_positions_for_grid 



def grid_data_lilearlog_for_1d(vis_obs, q_obs, wgt_obs, n_d_eq, n_d_log, xmin, xmax):
    
    coord_for_grid, rep_positions_for_grid = making_grids_nufft_combination_of_two_gridding_for_1d(xmax, xmin, n_d_eq, n_d_log)
    weight_sum, xedges, binnumber =  binned_statistic( q_obs, wgt_obs, 'sum', bins=coord_for_grid)
    vis_real_sum, xedges, binnumber =  binned_statistic(q_obs, vis_obs.real * wgt_obs, 'sum', bins=coord_for_grid)
    vis_imag_sum, xedges, binnumber =  binned_statistic(q_obs, vis_obs.imag * wgt_obs,  'sum', bins=coord_for_grid)
    mask = weight_sum>0
    vis_real_masked = np.ravel(vis_real_sum[mask])/np.ravel(weight_sum[mask])    
    vis_imag_masked = np.ravel(vis_imag_sum[mask])/np.ravel(weight_sum[mask])
    vis_grid_1d = vis_real_masked  + 1j * vis_imag_masked 
    weight_grid_1d = np.ravel(weight_sum[mask])
    noise_grid_1d = (1/weight_grid_1d**0.5) + 1j * (1/weight_grid_1d **0.5)
    q_grid_1d = np.ravel(rep_positions_for_grid[mask])
    return q_grid_1d, vis_grid_1d, noise_grid_1d


def making_grids_nufft_combination_of_two_gridding(xmax, xmin, n_d_eq, n_d_log):
    
    d_linear_x = xmax/n_d_eq
    x_max_for_log_grid = calc_boundary_bewteen_log_and_linear(xmax, xmin, n_d_eq, n_d_log)

    x_coord_linear = np.arange(x_max_for_log_grid, x_max_for_log_grid +  d_linear_x*n_d_eq, step = d_linear_x)
    x_coord_linear = x_coord_linear[x_coord_linear-d_linear_x<xmax]
    x_coord_linear = x_coord_linear[1:]
    xedges_for_meshgrid_side_linear = x_coord_linear - 0.5*d_linear_x 
        
    dx = (np.log10(x_max_for_log_grid) - np.log10(xmin))/(n_d_log+1.0)
    xedges = np.linspace(np.log10(xmin), np.log10(x_max_for_log_grid), n_d_log+1)
    xedges_for_meshgrid_side = 0.5*dx  + np.linspace(np.log10(xmin), np.log10(x_max_for_log_grid), n_d_log+1)
    xedges_for_meshgrid_side = xedges_for_meshgrid_side[:n_d_log]

    ##
    coord_for_grid = np.append(10**xedges, -10 ** xedges)
    coord_for_grid = np.append(coord_for_grid, 0)
    coord_for_grid = np.append(coord_for_grid, x_coord_linear)
    coord_for_grid = np.append(coord_for_grid, - x_coord_linear)
    coord_for_grid = np.sort(coord_for_grid)

    ##
    rep_positions_for_grid = np.append(10**xedges_for_meshgrid_side, -10 ** xedges_for_meshgrid_side)
    rep_positions_for_grid = np.append(rep_positions_for_grid, [-0.5 * xmin, 0.5 * xmin])
    rep_positions_for_grid = np.append(rep_positions_for_grid, xedges_for_meshgrid_side_linear)
    rep_positions_for_grid = np.append(rep_positions_for_grid, - xedges_for_meshgrid_side_linear)
    rep_positions_for_grid = np.sort(rep_positions_for_grid)
    uu_for_grid_pos, vv_for_grid_pos= np.meshgrid(rep_positions_for_grid, rep_positions_for_grid)
    return coord_for_grid, rep_positions_for_grid, uu_for_grid_pos, vv_for_grid_pos



def grid_2dvis_log10_edges_combination_of_two_gridding_using_weights(vis_obs, u_obs, v_obs, wgt_obs, n_d_eq, n_d_log, xmin, xmax):

    """
    Gridding multi-frequency visibility data

    arguments
    =========
    vis_obs: 2D numpy.ndarray (x_len, x_len)
        Exponent array for log10
    u_obs: 1D numpy.ndarray (n_obs)
        u_obs[i] is u-baseline (d_x[i]/lambda[i])
    v_obs: 1D numpy.ndarray (n_obs)
        v_obs[i] is v-baseline (d_y[i]/lambda[i])
    dx: float
        unit of x length of image
    dy: float
        unit of y length of image   
    n_freq: int
        number of frequency
    x_len: int
        x length of image
    y_len: int
        y length of image

    return
    ======
    vis_freq: 1D numpy.ndarray 
        gridded visibility
    num_mat_freq: 1D numpy.ndarray 
        number of measurements on each bin
    noise_freq: 1D numpy.ndarray 
        uncertainties for gridded visibility
    """


    ##
    coord_for_grid, rep_positions_for_grid, uu_for_grid_pos, vv_for_grid_pos = making_grids_nufft_combination_of_two_gridding(xmax, xmin, n_d_eq, n_d_log)

    weight_sum, xedges, yedges, binnumber = binned_statistic_2d(v_obs, u_obs, wgt_obs, statistic="sum", bins=[coord_for_grid, coord_for_grid])      
    vis_real_sum, xedges, yedges, binnumber = binned_statistic_2d(v_obs, u_obs, wgt_obs * vis_obs.real, statistic="sum", bins=[coord_for_grid, coord_for_grid])      
    vis_imag_sum, xedges, yedges, binnumber = binned_statistic_2d(v_obs, u_obs, wgt_obs *  vis_obs.imag, statistic="sum", bins=[coord_for_grid, coord_for_grid])      

    mask = weight_sum>0
    vis_real_masked = np.ravel(vis_real_sum[mask])/np.ravel(weight_sum[mask])
    vis_imag_masked = np.ravel(vis_imag_sum[mask])/np.ravel(weight_sum[mask])
    vis_grid_1d = vis_real_masked  + 1j * vis_imag_masked 
    weight_grid_1d = np.ravel(weight_sum[mask])
    u_grid_1d = np.ravel(uu_for_grid_pos[mask])
    v_grid_1d = np.ravel(vv_for_grid_pos[mask])
    noise_grid_1d = (1/weight_grid_1d**0.5) + 1j * (1/weight_grid_1d **0.5)

    return u_grid_1d, v_grid_1d, vis_grid_1d, noise_grid_1d








def load_and_grid_visdata(load_file = "concat_X13_17_selfcal.npz", x_len=512, save_file =None):
    if save_file is not None and os.path.exists(save_file):

        gridded_vis = np.load(save_file, allow_pickle=True)
        return gridded_vis["u_freq"], gridded_vis["v_freq"], gridded_vis["uv_stack"], gridded_vis["vis_freq"], gridded_vis["noise_freq"], gridded_vis["freq_arr"]

    vis_data = np.load(load_file, allow_pickle=True)
    u_freq = vis_data["u_freq"]
    v_freq = vis_data["v_freq"]
    vis_freq = vis_data["vis_freq"]
    nu_arr = vis_data["freq_arr"]

    x_min_arr = []
    x_max_arr = []

    for i in range(len(u_freq)):
        x_min_arr.append(0.1 * np.min( (u_freq[i]**2+v_freq[i]**2)**0.5))
        x_max_arr.append(1.1 * np.max(np.abs(v_freq[i])))
        x_max_arr.append(1.1 * np.max(np.abs(u_freq[i])))
    x_min = np.min(x_min_arr)
    x_max = np.max(x_max_arr)

    u_grid_freq, v_grid_freq, vis_grid_freq, noise_grid_freq, count_grid_freq = grid_2dvis_log10_edges_multifreq(vis_freq, u_freq, v_freq, x_len, x_min, x_max)
    uv_stack = []
    for i in range(len(u_grid_freq)):
        uv_stack_now = np.array([u_grid_freq[i], v_grid_freq[i]])
        uv_stack.append(uv_stack_now.T)
    uv_stack = np.array(uv_stack)
    if save_file is not None:
            np.savez(save_file.replace(".npz",""), uv_stack = uv_stack, u_freq = u_grid_freq, v_freq = v_grid_freq, vis_freq = vis_grid_freq, noise_freq = noise_grid_freq, freq_arr = nu_arr, count_grid_freq = count_grid_freq)
    return u_grid_freq, v_grid_freq, uv_stack, vis_grid_freq, noise_grid_freq, count_grid_freq, nu_arr
