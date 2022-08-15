import numpy as np
import os

def make_folder(folder):

    if not os.path.exists(folder):
        os.makedirs(folder)


def load_names(file):
    file = open(file, "r")
    lines = file.readlines()
    names = []
    for line in lines:
        name = line.replace("\n","").replace(" ", "")
        if len(name) ==0:
            continue
        names.append(name)
    return names

def make_vis_files(vis_folder, names):
    files = []
    for name in names:
        file_now = "averaged_%s_continuum.vis.npz" % name
        files.append(os.path.join(vis_folder, file_now))
    return files


def load_original_uv(file_name):
    """ Loading function for multi-freq data. This function deprojects visibility using 
    assumed inclination and positiona angle. 
    
    Args:
        file_name (str): file name for data
        inc (float): inclination of object
        pa (flaot): position angle of object
    
    Returns:
        q_dist (ndarray): 1d array containing visibility distance in radial direction
        vis_d_real (ndarray): 1d array contatining real part of visibility in radial direction
        freq_d (ndarray): 1d array containig frequency 
    """
    data = np.load(file_name)
    uvw = data["uvw_original"]
    u = uvw[0,:]
    v = uvw[1,:]
    return u, v


def load_obsdata(file_name):
    """ Loading function for multi-freq data. This function deprojects visibility using 
    assumed inclination and positiona angle. 
    
    Args:
        file_name (str): file name for data
        inc (float): inclination of object
        pa (flaot): position angle of object
    
    Returns:
        q_dist (ndarray): 1d array containing visibility distance in radial direction
        vis_d_real (ndarray): 1d array contatining real part of visibility in radial direction
        freq_d (ndarray): 1d array containig frequency 
    """
    data = np.load(file_name)
    v_d = data["v_obs"]
    u_d = data["u_obs"]
    vis_d = data["vis_obs"]
    wgt_d = data["wgt_obs"]
    freq_d = data["freq_obs"]
    return u_d, v_d, vis_d, wgt_d, freq_d

def q_max_determine(u_d, v_d):
    cosi_arr = np.linspace(0,1,21)
    pa_arr = np.linspace(0,np.pi,30) 
    q_max_arr = []

    for cosi_now in cosi_arr:
        for pa_now in pa_arr:

            cos_pa = np.cos(pa_now)
            sin_pa = np.sin(pa_now)
            u_new_d = -cos_pa * u_d + sin_pa *v_d
            v_new_d = -sin_pa * u_d - cos_pa *v_d
            u_new_d = cosi_now * u_new_d
            q_max = np.max(( (u_new_d)**2 + v_new_d**2)**0.5)
            q_max_arr.append(q_max)
    return np.min(q_max_arr)