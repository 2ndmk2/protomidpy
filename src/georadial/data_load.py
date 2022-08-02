import numpy as np
import os
from scipy.stats import binned_statistic_2d
import pandas as pd

def freqs_nu_file(file_freq):
    freqs = np.load(file_freq)
    return freqs["nu_arr"], freqs["nu0"]

def loader_of_visibility_from_csv_nonoise(vis_folder, nu_arr, config_ant = "7.6"):

    obs_vis = []
    u_obs = []
    v_obs = []
    vis_obs = []

    for nu_now in nu_arr:
        file = os.path.join(vis_folder, "psim_freq%d.alma.cycle%s.csv") % (int(nu_now), config_ant)
        df_none = pd.read_csv(file,\
         header=None)
        u = df_none[0]
        v = df_none[1]
        real = df_none[6]
        imag = -df_none[7] ## NOTE THAT CASA gives - imag
        u_obs.append(u)
        v_obs.append(v)
        vis_obs.append(real + 1j * imag)
    return np.array(u_obs), np.array(v_obs), np.array(vis_obs)
def loader_of_visibility_from_csv(vis_folder, nu_arr, config_ant = "7.6"):

    obs_vis = []
    u_obs = []
    v_obs = []
    vis_obs = []

    for nu_now in nu_arr:
        file = os.path.join(vis_folder, "psim_freq%d.alma.cycle%s.noisy.csv") % (int(nu_now), config_ant)
        df_none = pd.read_csv(file,\
         header=None)
        u = df_none[0]
        v = df_none[1]
        real = df_none[6]
        imag = -df_none[7] ## NOTE THAT CASA gives - imag
        u_obs.append(u)
        v_obs.append(v)
        vis_obs.append(real + 1j * imag)

    return np.array(u_obs), np.array(v_obs), np.array(vis_obs)

def loader_of_visibility_from_csv_from_singlefreq(file_name):

    obs_vis = []
    u_obs = []
    v_obs = []
    vis_obs = []

    df_none = pd.read_csv(file_name, header=None)
    u = df_none[0]
    v = df_none[1]
    real = df_none[6]
    imag = -df_none[7] ## NOTE THAT CASA gives - imag
    u_obs.append(u)
    v_obs.append(v)
    vis_obs.append(real + 1j * imag)

    return np.array(u_obs), np.array(v_obs), np.array(vis_obs)
def load_vis_from_npz(file_name):
    """
    Load visibility from numpy data

    arguments:
        file_name: numpy file that containts visibility data

    return:
        vis_obs (numpy.ndarray): 3D visibility data
        u_obs (numpy.ndarray): u_obs[i] is u for i-th frequency (d_x[i]/lambda[i])
        v_obs (numpy.ndarray): v_obs[i] is v-baseline (d_y[i]/lambda[i])
        freq (numpy.ndarray): 1D numpy.ndarray (nfreq)

    """
    data = np.load(file_name)
    u_obs=data["u_freq"]
    v_obs=data["v_freq"]
    vis_obs=data["vis_freq"]
    freq=data["freq_arr"]
    return vis_obs, u_obs, v_obs, freq


def grided_vis_from_obs(vis_obs, u_obs, v_obs, dx, dy, nfreq, x_len, y_len):
    """
    Gridding multi-frequency visibility data

    arguments
    =========
    vis_obs: 3D numpy.ndarray (nfreq, x_len, x_len)
        Exponent array for log10
    u_obs: 2D numpy.ndarray (nfreq, n_obs)
        u_obs[i] is u-baseline (d_x[i]/lambda[i])
    v_obs: 2D numpy.ndarray (nfreq, n_obs)
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
    vis_freq: 3D numpy.ndarray (nfreq, x_len, x_len)
        gridded visibility
    num_mat_freq: 3D numpy.ndarray (nfreq, x_len, x_len)
        number of measurements on each bin
    noise_freq: 3D numpy.ndarray (nfreq, x_len, x_len)
        uncertainties for gridded visibility
    """

    du = 1/(dx * x_len)
    dv = 1/(dx * y_len)
    u = np.arange(0,x_len * du, du)
    v = np.arange(0,y_len * dv, dv)
    u_shift = u - np.mean(u)
    v_shift = v - np.mean(v)
    for_bins_u = np.append(u_shift, np.max(u_shift)+du) - du/2
    for_bins_v = np.append(v_shift, np.max(v_shift)+dv) - dv/2

    vis_freq = np.zeros((nfreq, x_len, y_len), dtype=np.complex)
    noise_freq = np.zeros((nfreq, x_len, y_len), dtype=np.complex)
    num_mat_freq = np.zeros((nfreq, x_len, y_len), dtype=np.complex)
    print(np.max(u), np.max(u_obs))
    print(np.max(v), np.max(v_obs))

    for i_freq in range(nfreq):

        ret = binned_statistic_2d(\
            v_obs[i_freq], u_obs[i_freq], vis_obs[i_freq].real, statistic="mean", \
            bins=(for_bins_u, for_bins_v))
        mean_bins_real = ret.statistic

        ret = binned_statistic_2d(\
            v_obs[i_freq], u_obs[i_freq], vis_obs[i_freq].imag, statistic="mean", \
            bins=(for_bins_u, for_bins_v))
        mean_bins_imag = ret.statistic
        mean_bins = mean_bins_real + 1j * mean_bins_imag

        ret = binned_statistic_2d(\
            v_obs[i_freq],u_obs[i_freq],  vis_obs[i_freq].real, statistic="count", \
            bins=(for_bins_u, for_bins_v))
        num_mat = ret.statistic


        ret = binned_statistic_2d(\
            v_obs[i_freq], u_obs[i_freq], vis_obs[i_freq].real, statistic="std", \
            bins=(for_bins_u, for_bins_v))
        std_bins_real = ret.statistic


        ret = binned_statistic_2d(\
            v_obs[i_freq],u_obs[i_freq],  vis_obs[i_freq].imag, statistic="std", \
            bins=(for_bins_u, for_bins_v))
        std_bins_imag = ret.statistic
        std_bins = (std_bins_real + 1j * std_bins_imag)/(np.sqrt(num_mat-1))


        flag_at_least_two_counts = num_mat < 3
        mean_bins[flag_at_least_two_counts] = 0
        std_bins[flag_at_least_two_counts] = 0
        num_mat[flag_at_least_two_counts] = 0

        vis_freq[i_freq] = mean_bins
        noise_freq[i_freq] = std_bins_real/(np.sqrt(num_mat-1)) + 1j* std_bins_imag/(np.sqrt(num_mat-1))
        num_mat_freq[i_freq] = num_mat

    return vis_freq, num_mat_freq, noise_freq


