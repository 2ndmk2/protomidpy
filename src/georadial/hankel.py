import numpy as np
from scipy.special import j0, j1, jn_zeros, jv
ARCSEC_TO_RAD= 1/206265.0

def diag_multi(diag_sigma, mat):
    return (diag_sigma * mat.T).T

def make_collocation_points(R_out, N):
    """ Making collocation points for Hankel transformation

    Params:
        R_out (float): Upper limit on radial size of object 
        N (int): number of radial points in model
    Returns:
        r_n (ndarray): 1d radial positions of collocation points
        j_nN (ndarray): roots of the order 0 Bessel function of the first kind
        q_max (float): Maximum visibility distance
    """

    j_nk = jn_zeros(0, N + 1)
    j_nk, j_nN = j_nk[:-1], j_nk[-1]
    r_n = R_out * j_nk/j_nN
    q_n = j_nk/(2 * np.pi * R_out )
    q_max =j_nN /(2 * np.pi * R_out)
    return r_n, j_nN, q_max, q_n

def prepare(R_out, nrad, d, sigma_mat_diag, file_name =None, rewrite =True):

    r_n, jn, qmax, q_n = make_collocation_points(R_out, nrad)
    H_mat_model = make_hankel_matrix(q_n, R_out, nrad,  1)
    q_dist_2d_model = make_2d_mat_for_dist(q_n)
    N_d = len(d)
    r_dist = make_2d_mat_for_dist(r_n)
    d_A_minus1_d = np.sum(sigma_mat_diag* d * d)
    logdet_for_sigma_d = -np.sum(np.log(sigma_mat_diag))
    return r_n, jn, qmax, q_n, H_mat_model, q_dist_2d_model, N_d, r_dist, d_A_minus1_d, logdet_for_sigma_d 

def hankel_precomp(R_out, nrad, dpix, u_d, v_d, d, sigma_d, cosi, pa, delta_x, delta_y):

    r_n, jn, qmax, q_n = make_collocation_points(R_out, nrad)
    factor_all, r_pos =make_hankel_matrix_kataware( R_out, nrad, dpix)
    H_mat = make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, nrad, factor_all, r_pos, dpix,  qmax)
    V_A_minus1_U = H_mat.T@diag_multi(sigma_d, H_mat)
    V_A_minus1_d = H_mat.T@(sigma_d*d)
    return H_mat , V_A_minus1_U , V_A_minus1_d



def make_2d_mat_for_dist(r_arr):
    r_pos_tile = np.tile(r_arr,(len(r_arr),1)) 
    r_dist = ((r_pos_tile - r_pos_tile.T)**2)**0.5
    return r_dist


def make_hankel_matrix_kataware(R_out, N, dpix):
    j_nplus = jn_zeros(0, N+1)
    j_nk = jn_zeros(0, N + 1)
    j_nk, j_nN = j_nk[:-1], j_nk[-1]
    r_pos =2 * np.pi *  R_out * j_nk/j_nN
    factor = ( 1/ARCSEC_TO_RAD**2) * 4 * np.pi * R_out**2 / (j_nN**2)
    scale_factor = 1/(j1(j_nk) ** 2)
    q_max =j_nN /(2 * np.pi * R_out)
    scale_all = factor * scale_factor
    return  scale_all, r_pos

def make_hankel_matrix_from_kataware(q, scale_all, r_pos, q_max, cosi):
    H_mat = scale_all * j0(np.outer(q,r_pos))
    H_mat[q>q_max] = 0
    return cosi * H_mat

def make_hankel_matrix(q, R_out, N, cosi):
    j_nplus = jn_zeros(0, N+1)
    j_nk = jn_zeros(0, N + 1)
    j_nk, j_nN = j_nk[:-1], j_nk[-1]
    r_pos = R_out * j_nk/j_nN
    factor = ( 1/ARCSEC_TO_RAD**2) * 4 * np.pi * R_out**2 / (j_nN**2)
    scale_factor = 1/(j1(j_nk) ** 2)
    q_max =j_nN /(2 * np.pi * R_out)
    H_mat = factor * scale_factor * j0(np.outer(q, 2 * np.pi * r_pos))
    H_mat[q>q_max] = 0
    return cosi * H_mat


def make_hankel_wt_fixed_q(q_dist, R_out, N, factor_all, r_pos, dpix,q_max):
    H_mat = make_hankel_matrix_from_kataware(q_dist,factor_all, r_pos, q_max, 1)
    H_mat_all = np.concatenate([H_mat, np.zeros(np.shape(H_mat))])
    return H_mat_all

def make_hankel_at_inc_pa_w_offset(u_d, v_d, cosi, pa, delta_x, delta_y, R_out, N, factor_all, r_pos, dpix,q_max):
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    u_new_d = cos_pa * u_d - sin_pa *v_d
    v_new_d = sin_pa * u_d + cos_pa *v_d
    u_new_d = u_new_d * cosi
    q_dist = (u_new_d**2 + v_new_d **2)**0.5
    H_mat = make_hankel_matrix_from_kataware(q_dist,factor_all, r_pos, q_max, cosi)
    diag_mat_cos = np.cos(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat_sin = np.sin(2 * np.pi * (- delta_x * u_d - delta_y * v_d))
    diag_mat = np.append(diag_mat_cos, diag_mat_sin)
    H_mat_off1 = diag_multi(diag_mat_cos, H_mat)
    H_mat_off2 = diag_multi(diag_mat_sin, H_mat)
    H_mat_all = np.concatenate([H_mat_off1, H_mat_off2])
    return H_mat_all

def model_for_vis(I_model, q_dist, R_out, N, dpix, cosi):
    H_mat = make_hankel_matrix(q_dist, R_out, N, cosi)
    vis_model = np.dot(H_mat, I_model)
    return vis_model