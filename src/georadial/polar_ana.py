
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import LinearNDInterpolator
import pickle
from georadial import data_gridding


def get_image(target_name, folder_image):

    image_file = os.path.join(folder_image,"image_%s.fits" % target_name )
    if not os.path.exists(image_file ):
        return None

    hdul = fits.open(image_file )
    image = hdul[0].data[0][0]
    hdul.close()
    return image

def get_image_and_res(target_name, folder_raw, folder_image):

    subtracted_file = os.path.join(folder_image,"image_%s.fits" % target_name )
    raw_file = os.path.join(folder_raw,"image_%s.fits" % target_name )
    if not os.path.exists(subtracted_file):
        return None, None
    if not os.path.exists(raw_file):
        return None, None

    hdul = fits.open(subtracted_file)
    hdul_raw = fits.open(raw_file)
    image_subtracted = hdul[0].data[0][0]
    image_raw =  hdul_raw[0].data[0][0]
    hdul.close()
    hdul_raw.close()
    return image_raw, image_subtracted

def make_coordinate(nx, dx):
    x = np.linspace(0, (nx-1)*dx, nx)  
    x -= np.mean(x)
    return x

def get_pa_coi(target_name, folder):
    mcmcfile = os.path.join(folder,"%s_continuum_averagedmodel.npz" % target_name )
    sample = np.load(mcmcfile)["sample_best"]        
    cosi = sample[2]
    pa = sample[3]#0.5 * np.pi - sample[3]# + 3 * np.pi/2
    return cosi, pa 

def interpolate_image(image, dx = 0.01, dy= 0.01):
    ## 
    nx, ny = np.shape(image)
    x = make_coordinate(nx, dx)
    y = make_coordinate(ny, dy)
    f = interpolate.interp2d(x, y, image, kind='cubic')
    return f

def interpolate_image_ND(image, dx = 0.01, dy= 0.01):
    ## 
    nx, ny = np.shape(image)
    x = make_coordinate(nx, dx)
    y = make_coordinate(ny, dy)
    xx, yy = np.meshgrid(x, y)
    xx = -xx
    f = LinearNDInterpolator(list(zip(np.ravel(xx), np.ravel(yy))), np.ravel(image))    
    return f

def make_interpolated_image_plus_ND(f, x_coord, y_coord, cosi, pa):
    xx_new, yy_new = np.meshgrid(x_coord, y_coord)
    xx_new = - xx_new
    xx2_new = xx_new   * cosi
    yy2_new = yy_new 
    xx3_new = xx2_new * np.cos(pa) + yy2_new * np.sin(pa)
    yy3_new =  -xx2_new * np.sin(pa) + yy2_new * np.cos(pa)
    imsize = len(x_coord)
    image_out = f(xx3_new, yy3_new)#np.zeros((imsize, imsize))

    return image_out, xx_new, yy_new


def main_interpolated_image(pickle_name, pickle_name2, res, dx_original_image, x_coord, cosi, pa):

    if not os.path.exists(pickle_name):
        int_f = interpolate_image_ND(res, dx_original_image, dx_original_image)
        with open(pickle_name, "wb") as f:
            pickle.dump(int_f, f)    
    else:
        with open(pickle_name, "rb") as f:
            int_f  = pickle.load(f)
    if not os.path.exists(pickle_name2):
        image_out, xx_new, yy_new = make_interpolated_image_plus_ND(int_f, x_coord, x_coord, cosi, pa)
        interpolator = LinearNDInterpolator(list(zip(np.ravel(xx_new), np.ravel(yy_new))), np.ravel(image_out))        

        with open(pickle_name2, "wb") as f:
            pickle.dump(interpolator, f)    
    else:
        with open(pickle_name2, "rb") as f:
            interpolator  = pickle.load(f)
            image_out, xx_new, yy_new = make_interpolated_image_plus_ND(interpolator, x_coord, x_coord, 1, 0)

    return interpolator, image_out

def polar_max(r_arr, phi, interpolator):
    abs_val_arr= []
    angle_arr = []
    for r_now in r_arr:
        z = interpolator(r_now * np.cos(phi ), r_now*np.sin(phi ))
        arg_max = np.argmax(z)
        abs_val_arr.append(z[arg_max])
        angle_arr.append(phi[arg_max])
    abs_val_arr = np.array(abs_val_arr)
    angle_arr = np.array(angle_arr)
    return abs_val_arr, angle_arr

def integral_polar(r_arr, phi, m, delta_phi, interpolator):

    abs_val_arr= []
    angle_arr = []
    for r_now in r_arr:
        z = interpolator(r_now * np.cos(phi ), r_now*np.sin(phi ))
        cosphi= np.cos(phi * m)
        sinphi= np.sin(phi * m)
        cos_value = np.sum(cosphi * z) * delta_phi
        sin_value = - np.sum(sinphi * z) * delta_phi
        angle = np.arctan2(sin_value , cos_value)
        abs_value = (1/np.pi) * (cos_value**2 + sin_value**2)**0.5
        abs_val_arr.append(abs_value)
        angle_arr.append(angle)
    abs_val_arr = np.array(abs_val_arr)
    angle_arr = np.array(angle_arr)
    return abs_val_arr, angle_arr

def integral_polar_shifted(r_arr, phi, m, delta_phi, dx, dy,  interpolator):

    abs_val_arr= []
    angle_arr = []
    for r_now in r_arr:
        z = interpolator(dx + r_now * np.cos(phi ), dy + r_now*np.sin(phi ))
        cosphi= np.cos(phi * m)
        sinphi= np.sin(phi * m)
        cos_value = np.sum(cosphi * z) * delta_phi 
        sin_value = -np.sum(sinphi * z) * delta_phi 
        angle = np.arctan2(sin_value , cos_value)
        abs_value = (1/np.pi) * (cos_value**2 + sin_value**2)**0.5
        abs_val_arr.append(abs_value)
        angle_arr.append(angle)
    abs_val_arr = np.array(abs_val_arr)
    angle_arr = np.array(angle_arr)
    return abs_val_arr, angle_arr




def make_cirle(rad_arr, cen, color="r"):
    circle_arr = []
    for rad in rad_arr:
        circle = plt.Circle((cen, cen), rad, fill=False, color= color)
        circle_arr.append(circle)
    return circle_arr


def make_interpolation_for_rad_profile(r_arr, flux_arr, r_space = 0.05):

    interp_f = interpolate.interp1d(r_arr, flux_arr, kind='cubic', fill_value="extrapolate")
                                       
    try:
        r_max_interp = np.min(r_arr[flux_arr<0])-r_space
        
    except:
        r_max_interp = np.max(r_arr)-r_space
    return r_max_interp, interp_f

def load_dsharp_profile(target_name, folder_name):    
    filename = os.path.join(folder_name , "%s.profile.txt" % target_name)
    data = np.loadtxt(filename)
    r_arr = data[:,1]
    flux_arr = data[:,2]
    r_max_interp, interp_f = make_interpolation_for_rad_profile(r_arr, flux_arr)

    return r_max_interp, interp_f

def make_radial_profile_main(image, dx,  cosi, pa):
    """ Module or making radial profile for disks
    
    """
    nx, ny = np.shape(image)
    x_coord = make_coordinate(nx, dx)
    xx_new, yy_new = np.meshgrid(x_coord, x_coord)
    xx2_new = xx_new * np.cos(pa) - yy_new * np.sin(pa)
    yy2_new = +xx_new * np.sin(pa) + yy_new * np.cos(pa)    
    xx3_new = xx2_new / cosi
    yy3_new = yy2_new
    rr_new = (xx3_new**2 + yy3_new**2 )**0.5
    image_1d = np.ravel(image)
    rr_1d = np.ravel(rr_new )

    return rr_1d, image_1d

def make_radial_profile_target(target_name, image_folder, mcmc_folder, dx, rmax = 2, n_d_log = 200):
    """ Make radial profile for target with name=target_name

    
    """

    image = get_image(target_name, image_folder)
    cosi, pa = get_pa_coi(target_name, mcmc_folder)
    nx, ny = np.shape(image)
    x_coord = make_coordinate(nx, dx)
    xx_new, yy_new = np.meshgrid(x_coord, x_coord)
    xx2_new = xx_new * np.cos(pa) - yy_new * np.sin(pa)
    yy2_new = +xx_new * np.sin(pa) + yy_new * np.cos(pa)    
    xx3_new = xx2_new / cosi
    yy3_new = yy2_new
    rr_new = (xx3_new**2 + yy3_new**2 )**0.5
    image_1d = np.ravel(image)
    rr_1d = np.ravel(rr_new )
    n_d_log = 200
    x_grid = data_gridding.linear_gridding_1d(0, rmax , n_d_log )
    r_1d, flux_1d, noise_1d, d_data, sigma_mat_1d = \
    data_gridding.data_binning_1d(rr_1d, image_1d, np.ones(np.shape(rr_1d)), x_grid[1:])
    r_max_interp, interp_f = make_interpolation_for_rad_profile(r_1d, flux_1d)

    return r_max_interp, interp_f 

"""
def make_interpolated_image(f, x_coord, y_coord, cosi, pa):
    x_coord_inv = x_coord
    xx_new, yy_new = np.meshgrid(y_coord, x_coord_inv)
    xx2_new = xx_new * cosi
    yy2_new = yy_new  
    xx3_new = xx2_new * np.cos(pa) + yy2_new * np.sin(pa)
    yy3_new =  -xx2_new * np.sin(pa) + yy2_new * np.cos(pa)
    imsize = len(x_coord)
    image_out = np.zeros((imsize, imsize))
    for i in range(imsize):
        for j in range(imsize):
            image_out[i][j] = f(xx3_new[i,j], yy3_new[i,j] )
    return image_out

def make_interpolated_image_plus(f, x_coord, y_coord, cosi, pa):
    x_coord_inv = x_coord
    xx_new, yy_new = np.meshgrid(y_coord, x_coord_inv)
    xx2_new = xx_new* cosi
    yy2_new = yy_new  
    xx3_new = xx2_new * np.cos(pa) + yy2_new * np.sin(pa)
    yy3_new =  -xx2_new * np.sin(pa) + yy2_new * np.cos(pa)
    imsize = len(x_coord)
    image_out = np.zeros((imsize, imsize))

    for i in range(imsize):
        for j in range(imsize):
            image_out[i][j] = f(xx3_new[i,j], yy3_new[i,j] )
    return image_out, xx_new, yy_new
"""
