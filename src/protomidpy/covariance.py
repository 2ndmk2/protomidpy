import numpy as np

ARCSEC_TO_RAD= 1/206265.0


def covariance_return(cov, theta, r_dist= None, q_dist_model = None,  H_mat_model = None,H_mat = None):

    if cov=="matern_q":
        K_cov, K_cov_inv = matern_for_k_space(q_dist_model,theta[1], theta[0], H_mat_model, nu = -100)
    if cov=="RBF_q":
        K_cov, K_cov_inv = RBF_for_k_space(q_dist_model,theta[1], theta[0], H_mat_model)
    if cov=="matern":
        K_cov, K_cov_inv = cov_matern(r_dist,theta, nu = -100)
    if cov=="RBF":
        K_cov, K_cov_inv = RBF_add_noise(r_dist,theta )
    if cov=="RBF_double":
        K_cov, K_cov_inv = RBF_add_noise_double(r_dist,theta )
    if cov=="power":
        K_cov, K_cov_inv = cov_power(q_dist,theta[2], theta[0], theta[1], H_mat)

    return K_cov, K_cov_inv 



def zero_for_cov(K_inv, n_index):

    K_inv[:n_index] = 0
    K_inv[:,:n_index] = 0
    return K_inv

def matern_for_k_space(obst, alpha, gamma, H_mat, q_constrained=0, nu = -100):
    """Compute precision matrix for visibility considering powerlaw typye power spectrum

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        alpha (float): value determines the height of power spectrun at q0 (minimum visibility distance)
        gamma (float): power law index (q_k/q_0)**(-gamma)
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        K_cov_inv (ndarray): 2d array for precision matrix for visibility
        logdet_cov (float): determinant of covariance matrix
    """

    Dt = obst
    nx, ny = np.shape(obst)
    sigma = 0

    K = matern(Dt, alpha, gamma, nu = nu)
    K_inv= np.linalg.inv(K)
    K_inv =  (zero_for_cov(K_inv, q_constrained ))# +  1e-2* np.identity(nx))
    inv_K_q = H_mat.T@K_inv@H_mat


    return None, inv_K_q


def RBF_for_k_space(obst, alpha, gamma, H_mat, q_constrained=0 , simga = 1e-10):
    """Compute precision matrix for visibility considering powerlaw typye power spectrum


    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        alpha (float): value determines the height of power spectrun at q0 (minimum visibility distance)
        gamma (float): power law index (q_k/q_0)**(-gamma)
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        K_cov_inv (ndarray): 2d array for precision matrix for visibility
        logdet_cov (float): determinant of covariance matrix
    """

    Dt = obst
    nx, ny = np.shape(obst)
    sigma = 0
    K=alpha * (np.exp(-(Dt/gamma)**2/2) + simga * np.identity(nx))
    #K=alpha * (np.exp(-(Dt/gamma)**2/2))# + simga * np.identity(nx))
    K_inv= np.linalg.inv(K)
    K_inv =  (zero_for_cov(K_inv, q_constrained ))# +  1e-2* np.identity(nx))
    inv_K_q = H_mat.T@K_inv@H_mat


    return None, inv_K_q

def TSV_for_k_space(obst, alpha, gamma, H_mat, simga = 1e-7):
    """Compute precision matrix for visibility considering powerlaw typye power spectrum

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        alpha (float): value determines the height of power spectrun at q0 (minimum visibility distance)
        gamma (float): power law index (q_k/q_0)**(-gamma)
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        K_cov_inv (ndarray): 2d array for precision matrix for visibility
        logdet_cov (float): determinant of covariance matrix
    """

    Dt = obst
    nx, ny = np.shape(obst)
    sigma = 0
    K=alpha * (np.exp(-(Dt/gamma)**2/2) + simga * np.identity(nx))
    K_inv= np.linalg.inv(K)
    K_inv =  (zero_for_cov(K_inv, 20))# +  1e-2* np.identity(nx))
    inv_K_q = H_mat.T@K_inv@H_mat

    return None, inv_K_q


def cov_power(q_n, a_power, alpha, gamma, H_mat, q0=10**4):
    """Compute precision matrix for visibility considering powerlaw typye power spectrum

    Args:
        q_n (ndarray): 1d array for visibility distance for model. Number points are same as radial points for model. 
        alpha (float): value determines the height of power spectrun at q0 (minimum visibility distance)
        gamma (float): power law index (q_k/q_0)**(-gamma)
        H_mat (ndarray): 2d array for design matrix converting intensity to visibility
    Returns:
        K_cov_inv (ndarray): 2d array for precision matrix for visibility
        logdet_cov (float): determinant of covariance matrix
    """
    power_qk = a_power * (q_n/q0)**(-gamma)
    power_diag = (1/alpha) * np.diag(1/power_qk**2)
    K_cov_inv = H_mat.T@power_diag@H_mat
    (sign, logdet_inv_cov) = np.linalg.slogdet(K_cov_inv)
    logdet_cov = 1/logdet_inv_cov
    return K_cov_inv, logdet_cov    

def matern(r_dist, alpha, gamma, nu=0.5, simga = 1e-10):
    nx, ny = np.shape(r_dist)
    ins = (np.sqrt(3)* r_dist/gamma)    

    if nu == -100:
        K= alpha * ( (1/(1 + (r_dist/gamma)**2)) + simga * np.identity(nx))
    if nu==0.5:
        K= alpha *(np.exp(-ins/np.sqrt(3)) + simga * np.identity(nx))
    if nu==1.5:
        K= alpha *((1 + ins ) * np.exp(-ins) + simga * np.identity(nx))
    return K


def cov_matern(r_dist, theta, nu=-100):
    """ Calculate convariance matrix and its inverse

    Args:
        r_dist (ndarray): 2d array containing distnace between different pixels 
        theta (ndarray): 1d array for parameters.
    Returns:
        K_cov(ndarray): 2d convariance matrix for model prior. This is used as prior information. 
        K_cov_inv(ndarray): 2d precision matrix for model prior. This is used as prior information. 

    """
    gamma = theta[0]
    alpha = theta[1]
    K_cov = matern(r_dist, alpha, gamma, nu = nu)
    K_cov_inv= np.linalg.inv(K_cov)
    return K_cov, K_cov_inv


def RBF_add_noise(obst, theta, simga = 1e-8):
    alpha = 10.0**theta[1]
    gamma= ARCSEC_TO_RAD*theta[0]
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    nx, ny = np.shape(obst)
    K=alpha * (np.exp(-(Dt/gamma)**2/2) + simga * np.identity(nx))
    K_inv= np.linalg.inv(K)

    return K, K_inv

def RBF_add_noise_double(obst, theta, simga = 1e-8):
    alpha = 10.0**theta[1]
    gamma= ARCSEC_TO_RAD*theta[0]
    gamma2= ARCSEC_TO_RAD*theta[6]
    print(theta[0], theta[6])
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    nx, ny = np.shape(obst)
    K=alpha * (np.exp(-(Dt/gamma)**2/2) + np.exp(-(Dt/gamma2)**2/2)  + simga * np.identity(nx))
    K_inv= np.linalg.inv(K)

    return K, K_inv
