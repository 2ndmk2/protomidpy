
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
import matplotlib.patches as mpatches
from math import floor, log10

def sci_notation(num, decimal_digits=2, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.

    Taken from https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)

def compute_beamarea_from_fits(file):
    
    hdul = fits.open(file)
    header = hdul[0].header
    bmaj = header["BMAJ"] * 3600
    bmin = header["BMIN"] * 3600
    bpa =  header["BPA"]
    beam_area = (np.pi/(4*np.log(2)))*bmaj * bmin
    return bmaj, bmin, bpa, beam_area



def get_image(target_name, folder_image):

    image_file = os.path.join(folder_image,"image_%s.fits" % target_name )
    if not os.path.exists(image_file ):
        return None

    hdul = fits.open(image_file )
    image = np.flip(hdul[0].data[0][0], axis=1)# * 4/3
    hdul.close()
    return image

def get_image_from_file(image_file):

    if not os.path.exists(image_file ):
        return None

    hdul = fits.open(image_file )
    image =  np.flip(hdul[0].data[0][0], axis=1)
    hdul.close()
    return image

def make_coordinate(nx, dx):
    x = np.linspace(0, (nx-1)*dx, nx)  
    x -= np.mean(x)
    return x

def get_pa_cosi(target_name, folder):
    mcmcfile = os.path.join(folder,"%s_continuum_averagedmodel.npz" % target_name )
    sample = np.load(mcmcfile)["sample_best"]        
    cosi = sample[2]
    pa = sample[3]#0.5 * np.pi - sample[3]# + 3 * np.pi/2
    return cosi, pa 


def interpolate_image_ND(image, dpix = 0.01):
    ## 
    nx, ny = np.shape(image)
    x = make_coordinate(nx, dpix)
    y = make_coordinate(ny, dpix)
    xx, yy = np.meshgrid(x, y)
    f = LinearNDInterpolator(list(zip(np.ravel(xx), np.ravel(yy))), np.ravel(image))    
    return f

def make_interpolated_deprojected_image(f, x_coord, y_coord, cosi, pa):
    xx_new, yy_new = np.meshgrid(x_coord, y_coord)
    xx_new = - xx_new
    xx2_new = xx_new   * cosi
    yy2_new = yy_new 
    xx3_new = xx2_new * np.cos(pa) + yy2_new * np.sin(pa)
    yy3_new =  -xx2_new * np.sin(pa) + yy2_new * np.cos(pa)
    imsize = len(x_coord)
    image_out = f(xx3_new, yy3_new)#np.zeros((imsize, imsize))
    return image_out, xx_new, yy_new

def make_polar_rad_interpolated_image(f, r_1d, polar_1d):

    rr_new, polar_new = np.meshgrid(r_1d, polar_1d)
    xx = rr_new * np.cos(polar_new)
    yy = rr_new * np.sin(polar_new)
    image_out = f(xx, yy)
    return image_out

def make_sym_antisym_geoimage(f, x, y):

    xx, yy = np.meshgrid(x, y)
    image_sym = 0.5 * (f(xx, yy) + f(xx, -yy))
    image_antisym = 0.5 * (f(xx, yy) - f(xx, -yy))
    return image_sym, image_antisym



def make_interpolated_pickle(pickle_name, res, dx_original_image):

    if not os.path.exists(pickle_name):
        int_f = interpolate_image_ND(res, dx_original_image)
        with open(pickle_name, "wb") as f:
            pickle.dump(int_f, f)    
    else:
        with open(pickle_name, "rb") as f:
            int_f  = pickle.load(f)
            
    return int_f


def main_interpolated_image(pickle_name, pickle_name2, res, dx_original_image, x_coord, cosi, pa):

    if not os.path.exists(pickle_name):
        int_f = interpolate_image_ND(res, dx_original_image)
        with open(pickle_name, "wb") as f:
            pickle.dump(int_f, f)    
    else:
        with open(pickle_name, "rb") as f:
            int_f  = pickle.load(f)

    if not os.path.exists(pickle_name2):
        image_out, xx_new, yy_new = make_interpolated_deprojected_image(int_f, x_coord, x_coord, cosi, pa)
        interpolator = LinearNDInterpolator(list(zip(np.ravel(xx_new), np.ravel(yy_new))), np.ravel(image_out))        

        with open(pickle_name2, "wb") as f:
            pickle.dump(interpolator, f)    
    else:
        with open(pickle_name2, "rb") as f:
            interpolator  = pickle.load(f)
            image_out, xx_new, yy_new = make_interpolated_deprojected_image(interpolator, x_coord, x_coord, 1, 0)

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
        sin_value =  np.sum(sinphi * z) * delta_phi
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


def make_cirle(rad_arr, cen, color="r", linestyle ="-", lw = 2, alpha = 0.5):
    circle_arr = []
    for rad in rad_arr:
        circle = plt.Circle((cen, cen), rad, fill=False, color= color, linestyle  = linestyle, lw = lw,  alpha = alpha )
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

def load_dsharp_profile_obs(target_name, folder_name):    
    filename = os.path.join(folder_name , "%s.profile.txt" % target_name)
    data = np.loadtxt(filename)
    au_arr = data[:,0]
    r_arr = data[:,1]
    flux_arr = data[:,2]

    return au_arr, r_arr, flux_arr


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

def load_radial_profile(target_name, folder):
    """
    Get radial positions & profiles
    Params:
        target_name: object name
        folder: containing folders for profiles
    Return:
        r_arr: array for radial positions
        flux_arr:  array for flux

    """
    data = np.loadtxt(os.path.join(folder, "%s.profile.txt" % target_name))
    r_arr = data[:,1]
    flux_arr = data[:,2]
    print(np.shape(data))

    return r_arr, flux_arr

def load_radmax(target_name, file ="./size.dat"):
    lines = open(file, "r").readlines()
    dic = {}
    for line in lines:
        itemList = line.split()
        dic[itemList[0]] = float(itemList[1])
    dic["flagged_vis_for_doar25"]=1.2
    dic["flagged_vis_for_doar25_high_freq"]=1.2
    if target_name in dic:
        return dic[target_name]
    else:
        return 1.0

def make_radial_profile_target(target_name, image_folder, mcmc_folder, dx, rmax = 2, n_d_log = 200):
    """ Make radial profile for target with name=target_name
    
    """
    image = get_image(target_name, image_folder)
    cosi, pa = get_pa_cosi(target_name, mcmc_folder)
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

def std_computation_for_image(image_res, r_max, dpix):
    """
    Compute image for region w/ r>r_max

    Params:
        image_res: residual image
        r_max: boundary for comupting std
        dpix: pixel scale for image
    Return:
        None
    """
    nx, ny = np.shape(image_res)
    x = make_coordinate(nx, dpix)
    y = make_coordinate(ny, dpix)
    xx, yy = np.meshgrid(x, y)
    rr = (xx**2 + yy**2)**0.5
    mask_is_finite = np.isfinite(image_res)
    mask_rmax_out = rr >r_max
    res_std = np.std(image_res[mask_rmax_out * mask_is_finite])
    return res_std


def make_deprojected_image(target_name, folder_mcmc,pickle_name_before_deprojected, pickle_name_deprojected, res, dx_image, pix_interp, dx_interp ):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_mcmc: folder containing mcmc result
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        res: residual image (N * N)
        dx_image: pixel scale for image
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
    Return:
        interpolator: Interpolating function for images ( f(x,y))
        image_out: Interpolated imagme (N*N)

    """

    x_coord = make_coordinate(pix_interp, dx_interp)
    cosi, pa = get_pa_cosi(target_name,folder_mcmc)
    interpolator, image_out = main_interpolated_image(pickle_name_before_deprojected, pickle_name_deprojected, res, dx_image, x_coord, cosi, pa)

    return interpolator, image_out


def plotter_2_2_image(target_name, folder_raw, folder_res, folder_mcmc, folder_fig, pickle_name_before_deprojected, pickle_name_deprojected, \
     max_std_lw, max_std_up, pix_interp = 500, dx_interp = 0.01, rad_circle_arr = [0.25, 0.50, 1, 1.5], cen_circle =0, \
        dx_image = 0.006, cen = 0, plot_pix_image_scale = 1, plot_pix_image_scale_for_zoom = 0.5, cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )
        folder_mcmc: folder containing mcmc result
        folder_fig: folder for output fig
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        max_std_lw, max_std_up: lower, upper bounds for plots 
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
        rad_circle_arr: radii for circles plotted for reference
        cen_circle: center of circle
        dx_image: pixel scale for image
        cen: central coordinate of image
        plot_pix_image_scale: scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
        plot_pix_image_scale_for_zoom: zooming scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
    Return:
        None
    """

    ## Preparation
    image = get_image(target_name, folder_raw)
    res = get_image(target_name, folder_res)
    if image is None:
        return None, None

    r_max_interp, interp_f = make_radial_profile_target(target_name, folder_raw, folder_mcmc, dx_image)
    plot_pix_image = plot_pix_image_scale * r_max_interp
    plot_pix_image_zoom =plot_pix_image_scale_for_zoom  * r_max_interp
    interp_ext = dx_interp * pix_interp * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    nx, ny = np.shape(image)
    d_extent = dx_image * (nx/2)
    extent=[d_extent,-d_extent,+d_extent,-d_extent]
    std = np.std(res)

    interpolator, image_out = make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected, \
        pickle_name_deprojected, res, dx_image, pix_interp, dx_interp)


def plotter_2_2_image(target_name, folder_raw, folder_res, folder_mcmc, folder_fig, pickle_name_before_deprojected, pickle_name_deprojected, \
     max_std_lw, max_std_up, pix_interp = 500, dx_interp = 0.01, rad_circle_arr = [0.25, 0.50, 1, 1.5], cen_circle =0, \
        dx_image = 0.006, cen = 0, plot_pix_image_scale = 1, plot_pix_image_scale_for_zoom = 0.5, cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )
        folder_mcmc: folder containing mcmc result
        folder_fig: folder for output fig
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        max_std_lw, max_std_up: lower, upper bounds for plots 
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
        rad_circle_arr: radii for circles plotted for reference
        cen_circle: center of circle
        dx_image: pixel scale for image
        cen: central coordinate of image
        plot_pix_image_scale: scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
        plot_pix_image_scale_for_zoom: zooming scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
    Return:
        None
    """

    ## Preparation
    image = get_image(target_name, folder_raw)
    res = get_image(target_name, folder_res)    
    if image is None:
        return None, None

    r_max_interp, interp_f = make_radial_profile_target(target_name, folder_raw, folder_mcmc, dx_image)
    plot_pix_image = plot_pix_image_scale * r_max_interp
    plot_pix_image_zoom =plot_pix_image_scale_for_zoom  * r_max_interp
    interp_ext = dx_interp * pix_interp * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    nx, ny = np.shape(image)
    d_extent = dx_image * (nx/2)
    extent=[d_extent,-d_extent,+d_extent,-d_extent]
    std = np.std(res)

    interpolator, image_out = make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected, \
        pickle_name_deprojected, res, dx_image, pix_interp, dx_interp)



    ## Plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    z0 = axs[0,0].imshow(image**0.5, extent = extent, cmap = "jet")#, vmin = -10, vmax=10)
    axs[0,0].scatter(cen, cen, s = 10, color="k")
    #_cs2 = axs[0,0].contour(res, levels=[-max_std_lw * std, max_std_up * std], colors=["b","r"], alpha = 0.5)
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(z0,cax=cax)    
    axs[0,0].set_xlim(cen+plot_pix_image, cen-plot_pix_image)
    axs[0,0].set_ylim(cen-plot_pix_image, cen+plot_pix_image)    
    axs[0,0].set_title(" Image (%s)" % target_name)    
    axs[0,1].set_xlim(cen+plot_pix_image, cen-plot_pix_image)
    axs[0,1].set_ylim(cen-plot_pix_image, cen+plot_pix_image)
    axs[0,1].set_title("Residual (%s)" % target_name)    
    z1 = axs[0,1].imshow(res, vmin = -max_std_lw * std, vmax=max_std_up* std, extent = extent, cmap = cmap)
    axs[0,1].scatter(cen, cen, s = 10, color="r")
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(z1,cax=cax)
    divider0 = make_axes_locatable(axs[1,0])
    divider1 = make_axes_locatable(axs[1,1])  
    axs[1,0].set_title("Deprojected Residual")    
    axs[1,1].set_title("Deprojected Residual (zoom)")    

    circle_arr = make_cirle(rad_circle_arr , cen_circle )
    axs[1,0].set_xlim(cen_circle +plot_pix_image, cen_circle -plot_pix_image)
    axs[1,0].set_ylim(cen_circle-plot_pix_image, cen_circle +plot_pix_image)
    axs[1,0].scatter(cen_circle, cen_circle, s = 10, color="r")
    z0 = axs[1,0].imshow(image_out, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap = cmap)
    for circle_now in circle_arr:
        axs[1,0].add_patch(circle_now) 
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(z0,cax=cax0)
    
    circle_arr = make_cirle(rad_circle_arr , cen_circle)
    axs[1,1].set_xlim(cen_circle+plot_pix_image_zoom, cen_circle-plot_pix_image_zoom)
    axs[1,1].set_ylim(cen_circle-plot_pix_image_zoom, cen_circle+plot_pix_image_zoom)
    axs[1,1].scatter(cen_circle, cen_circle, s = 10, color="r")

    for circle_now in circle_arr:
        axs[1,1].add_patch(circle_now)
    z1 = axs[1,1].imshow(image_out, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap = cmap)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)    
    plt.colorbar(z1,cax=cax1)
    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()

    return None

def make_arr_pickle_names(target_name, pickle_folders):
    pickle_name_before_deprojected_arr = []
    pickle_name_deprojected_arr = []

    for pickle_folder in pickle_folders:
        
        pickle_name = os.path.join(pickle_folder,"%s_res_int.pickle" % target_name)
        pickle_name_deproject = os.path.join(pickle_folder,"%s_res_int_deproject.pickle" % target_name)
        pickle_name_before_deprojected_arr.append(pickle_name)
        pickle_name_deprojected_arr.append(pickle_name_deproject)

    return pickle_name_before_deprojected_arr, pickle_name_deprojected_arr

def plotter_res_real_imag(target_name, folder_raw, folder_res, folder_mcmc, folder_fig, pickle_name_before_deprojected_arr, pickle_name_deprojected_arr, \
     max_std_lw, max_std_up, pix_interp = 500, dx_interp = 0.01, rad_circle_arr = [0.25, 0.50, 1, 1.5], cen_circle =0, \
        dx_image = 0.006, cen = 0, plot_pix_image_scale = 1, plot_pix_image_scale_for_zoom = 0.5, cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )
        folder_mcmc: folder containing mcmc result
        folder_fig: folder for output fig
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        max_std_lw, max_std_up: lower, upper bounds for plots 
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
        rad_circle_arr: radii for circles plotted for reference
        cen_circle: center of circle
        dx_image: pixel scale for image
        cen: central coordinate of image
        plot_pix_image_scale: scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
        plot_pix_image_scale_for_zoom: zooming scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
    Return:
        None
    """

    ## Preparation
    image = get_image(target_name, folder_raw)
    res = get_image(target_name, folder_res)

    if image is None:
        return None, None
    r_max_interp, interp_f = make_radial_profile_target(target_name, folder_raw, folder_mcmc, dx_image)
    plot_pix_image = plot_pix_image_scale * r_max_interp
    plot_pix_image_zoom =plot_pix_image_scale_for_zoom  * r_max_interp
    interp_ext = dx_interp * pix_interp * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    d_extent = dx_image * (nx/2)
    extent=[d_extent,-d_extent,+d_extent,-d_extent]
    
    ## Plot
    fig, axs = plt.subplots(1, len(pickle_name_before_deprojected_arr), figsize=(20, 20))
    for i in range(len(pickle_name_before_deprojected_arr)):
        ax_now = axs[i]
        divider1 = make_axes_locatable(ax_now)  
        interpolator, image_out = make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected_arr[i], \
            pickle_name_deprojected_arr[i], None, dx_image, pix_interp, dx_interp)
        std = np.std(res)
        circle_arr = make_cirle(rad_circle_arr , cen_circle)
        ax_now.set_xlim(cen_circle+plot_pix_image, cen_circle-plot_pix_image)
        ax_now.set_ylim(cen_circle-plot_pix_image, cen_circle+plot_pix_image)
        ax_now.scatter(cen_circle, cen_circle, s = 10, color="r")
        for circle_now in circle_arr:
            ax_now.add_patch(circle_now)
        z1 = ax_now.imshow(image_out, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)    
        plt.colorbar(z1,cax=cax1)

    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return None

def plotter_2d_array(target_name, folder_raw, folder_res, folder_mcmc, folder_fig, pickle_name_before_deprojected_arr, pickle_name_deprojected_arr, \
     max_std_lw, max_std_up, pix_interp = 500, dx_interp = 0.01, rad_circle_arr = [0.25, 0.50, 1, 1.5], cen_circle =0, \
        dx_image = 0.006, cen = 0, plot_pix_image_scale = 1, plot_pix_image_scale_for_zoom = 0.5, nx_image = 2, ny_image = 2, cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )
        folder_mcmc: folder containing mcmc result
        folder_fig: folder for output fig
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        max_std_lw, max_std_up: lower, upper bounds for plots 
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
        rad_circle_arr: radii for circles plotted for reference
        cen_circle: center of circle
        dx_image: pixel scale for image
        cen: central coordinate of image
        plot_pix_image_scale: scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
        plot_pix_image_scale_for_zoom: zooming scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
    Return:
        None
    """

    ## Preparation
    image = get_image(target_name, folder_raw)
    res = get_image(target_name, folder_res)

    if image is None:
        return None, None
    r_max_interp, interp_f = make_radial_profile_target(target_name, folder_raw, folder_mcmc, dx_image)
    plot_pix_image = plot_pix_image_scale * r_max_interp
    plot_pix_image_zoom =plot_pix_image_scale_for_zoom  * r_max_interp
    interp_ext = dx_interp * pix_interp * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    extent_for_deprojected_cont=[interp_ext,-interp_ext,-interp_ext,+interp_ext]
    nx, ny = np.shape(image)
    d_extent = dx_image * (nx/2)
    extent=[d_extent,-d_extent,+d_extent,-d_extent]
    ## Plot
    fig, axs = plt.subplots( nx_image,  ny_image, figsize=(20, 20))
    for i in range( nx_image):
        for j in range( ny_image):
            num_now = i  + j* nx_image
            if num_now == len(pickle_name_deprojected_arr):
                break
            ax_now = axs[i, j]
            divider1 = make_axes_locatable(ax_now)  
            interpolator, image_out = make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected_arr[num_now], \
                pickle_name_deprojected_arr[num_now], None, dx_image, pix_interp, dx_interp)
            std = np.std(res)
            circle_arr = make_cirle(rad_circle_arr , cen_circle)
            ax_now.set_xlim(cen_circle+plot_pix_image, cen_circle-plot_pix_image)
            ax_now.set_ylim(cen_circle-plot_pix_image, cen_circle+plot_pix_image)
            #ax_now.scatter(cen_circle, cen_circle, s = 10, color="r")
            #for circle_now in circle_arr:
            #    ax_now.add_patch(circle_now)
            #z2 =ax_now.contour(image_out, levels = [-2 * std , 2 * std], colors=["w","k"], extent = extent_for_deprojected_cont  )
            z1 = ax_now.imshow(image_out, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )
            cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
            plt.colorbar(z1,cax=cax1)

    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return None

def plotter_basic(target_name, folder_raw, folder_res, folder_mcmc, folder_fig, pickle_name_before_deprojected, pickle_name_deprojected, \
     max_std_lw, max_std_up, pix_interp = 500, dx_interp = 0.01, r_arr_angle= None, angle_arr = None, rad_circle_dark_arr =  None, rad_circle_bright_arr = None, cen_circle =0, \
        dx_image = 0.006, cen = 0, plot_pix_image_scale = 1, plot_pix_image_scale_for_zoom = 0.5, title = "", cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )
        folder_mcmc: folder containing mcmc result
        folder_fig: folder for output fig
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        max_std_lw, max_std_up: lower, upper bounds for plots 
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
        rad_circle_dark_arr: radii for dark regions plotted for reference
        rad_circle_bright_arr: radii for bright regions plotted for reference
        cen_circle: center of circle
        dx_image: pixel scale for image
        cen: central coordinate of image
        plot_pix_image_scale: scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
        plot_pix_image_scale_for_zoom: zooming scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
    Return:
        None
    """

    ## Preparation
    image = get_image(target_name, folder_raw)
    res = get_image(target_name, folder_res)

    if image is None:
        return None, None
    r_max_interp, interp_f = make_radial_profile_target(target_name, folder_raw, folder_mcmc, dx_image)
    plot_pix_image = plot_pix_image_scale * r_max_interp
    plot_pix_image_zoom =plot_pix_image_scale_for_zoom  * r_max_interp
    interp_ext = dx_interp * pix_interp * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    extent_for_deprojected_cont=[interp_ext,-interp_ext,-interp_ext,+interp_ext]
    nx, ny = np.shape(image)
    d_extent = dx_image * (nx/2)
    extent=[d_extent,-d_extent,+d_extent,-d_extent]
    ## Plot
    fig, axs = plt.subplots(figsize=(10 , 10 ))
    ax_now = axs
    divider1 = make_axes_locatable(ax_now)  
    interpolator, image_out = make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected, \
        pickle_name_deprojected, None, dx_image, pix_interp, dx_interp)
    std = np.std(res)
    #print(std)
    ax_now.set_xlim(cen_circle+plot_pix_image, cen_circle-plot_pix_image)
    ax_now.set_ylim(cen_circle-plot_pix_image, cen_circle+plot_pix_image)
    ax_now.set_xlabel("$\Delta$ x [arcsec]")
    ax_now.set_ylabel("$\Delta$ y [arcsec]")

    if rad_circle_dark_arr is not None:
        circle_arr = make_cirle(rad_circle_dark_arr , cen_circle, color="g", linestyle ="dashed", lw = 3, alpha = 0.75)
        for circle in circle_arr:
            ax_now.add_patch(circle)

    if rad_circle_bright_arr is not None:
        circle_arr = make_cirle(rad_circle_bright_arr , cen_circle, color="k", linestyle ="dashed", lw = 3, alpha = 0.75)
        for circle in circle_arr:
            ax_now.add_patch(circle)

    ax_now.set_title(title)
    z1 = ax_now.imshow(image_out, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )
    cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
    cbar = plt.colorbar(z1,cax=cax1, ticks=[-max_std_lw * std , 0, max_std_up* std], label = "residual S/N")
    cbar.ax.set_yticklabels(["-%d" % max_std_lw, '0', "%d" % max_std_up]) 

    if r_arr_angle is not None:
        x = r_arr_angle * np.cos(angle_arr)
        y = -r_arr_angle * np.sin(angle_arr)
        ax_now.scatter(x, y, color ="tab:blue", alpha = 0.8)
        ax_now.scatter(-x, -y, color ="tab:blue", alpha = 0.8)

    plt.savefig(os.path.join(folder_fig, "spiral_detail_%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return None


def plotter_1d_array(target_name, folder_raw, folder_res, folder_mcmc, folder_fig, pickle_name_before_deprojected_arr, pickle_name_deprojected_arr, \
     max_std_lw, max_std_up, pix_interp = 500, dx_interp = 0.01, rad_circle_arr = [0.25, 0.50, 1, 1.5], cen_circle =0, \
        dx_image = 0.006, cen = 0, plot_pix_image_scale = 1, plot_pix_image_scale_for_zoom = 0.5, titles = [""], nx_image = 2, ny_image = 2, cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )
        folder_mcmc: folder containing mcmc result
        folder_fig: folder for output fig
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        max_std_lw, max_std_up: lower, upper bounds for plots 
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
        rad_circle_arr: radii for circles plotted for reference
        cen_circle: center of circle
        dx_image: pixel scale for image
        cen: central coordinate of image
        plot_pix_image_scale: scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
        plot_pix_image_scale_for_zoom: zooming scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
    Return:
        None
    """

    ## Preparation
    image = get_image(target_name, folder_raw)
    res = get_image(target_name, folder_res)

    if image is None:
        return None, None
    r_max_interp, interp_f = make_radial_profile_target(target_name, folder_raw, folder_mcmc, dx_image)
    plot_pix_image = plot_pix_image_scale * r_max_interp
    plot_pix_image_zoom =plot_pix_image_scale_for_zoom  * r_max_interp
    interp_ext = dx_interp * pix_interp * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    extent_for_deprojected_cont=[interp_ext,-interp_ext,-interp_ext,+interp_ext]
    nx, ny = np.shape(image)
    d_extent = dx_image * (nx/2)
    extent=[d_extent,-d_extent,+d_extent,-d_extent]
    ## Plot
    fig, axs = plt.subplots(nx_image, ny_image, figsize=(10 * ny_image, 10 * nx_image))
    for num_now in range(np.max([nx_image,ny_image])):
        ax_now = axs[num_now]
        divider1 = make_axes_locatable(ax_now)  
        interpolator, image_out = make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected_arr[num_now], \
            pickle_name_deprojected_arr[num_now], None, dx_image, pix_interp, dx_interp)
        std = np.std(res)
        #print(std)
        circle_arr = make_cirle(rad_circle_arr , cen_circle)
        ax_now.set_xlim(cen_circle+plot_pix_image, cen_circle-plot_pix_image)
        ax_now.set_ylim(cen_circle-plot_pix_image, cen_circle+plot_pix_image)
        if num_now ==0:
            ax_now.set_xlabel("$\Delta$ x [arcsec]")
            ax_now.set_ylabel("$\Delta$ y [arcsec]")
        else:
            ax_now.set_xticks([])
            ax_now.set_yticks([])            
        ax_now.set_title(titles[num_now])
        z1 = ax_now.imshow(image_out, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )
        cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
        cbar = plt.colorbar(z1,cax=cax1, ticks=[-max_std_lw * std , 0, max_std_up* std], label = "residual S/N")
        cbar.ax.set_yticklabels(["-%d" % max_std_lw, '0', "%d" % max_std_up])           

    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return None


def plotter_images_res_and_raw(target_name, folder_images, folder_fig,  \
     max_std_lw=5, max_std_up=5,  vlim = None,  dx_image = 0.006,  xlim_image = 1, titles = [""],  cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_images: folders for images (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        titles: array for titles
        cmap: type of color map
    Return:
        None
    """
    ny_image = len(folder_images)
    nx_image = 1

    ## Plot
    fig, axs = plt.subplots(nx_image, ny_image, figsize=(10 * ny_image, 10 * nx_image))
    for num_now in range(np.max([nx_image,ny_image])):
        ax_now = axs[num_now]
        divider1 = make_axes_locatable(ax_now)  
        res = get_image(target_name, folder_images[num_now])
        nx,_  = np.shape(res)
        interp_ext = dx_image * nx * 0.5
        extent_for_deprojected=[-interp_ext,+interp_ext,interp_ext,-interp_ext]
        std = np.std(res)

        bmaj, bmin, bpa, beam_area = compute_beamarea_from_fits(os.path.join(folder_images[num_now],"image_%s.fits" % target_name ))

        ax_now.set_xlim(xlim_image, -xlim_image)
        ax_now.set_ylim(-xlim_image, xlim_image)
        if num_now ==0:
            ax_now.set_xlabel("$\Delta x$ [arcsec]")
            ax_now.set_ylabel("$\Delta y$ [arcsec]")
        else:
            ax_now.set_xticks([])
            ax_now.set_yticks([])            
        ax_now.set_title(titles[num_now])
        if num_now ==0:
            z1 = ax_now.imshow(np.arcsinh(res*1500),  extent = extent_for_deprojected, cmap =  "inferno" )
        else:
            z1 = ax_now.imshow(res, vmin = vlim[0], vmax = vlim[1], extent = extent_for_deprojected, cmap =  cmap )
            if num_now==ny_image-1:
                cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
                cbar = plt.colorbar(z1,cax=cax1, label = "Intensity [Jy/beam]")
                #cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
                #cbar.ax.set_yticklabels(["-%d" % max_std_lw, '0', "%d" % max_std_up])    
            patch = mpatches.Ellipse([0.8 * xlim_image, -0.8 * xlim_image], float(bmaj), float(bmin), 90-float(bpa), fc='none', ls='solid', ec='k', lw=2.) 
            ax_now.add_patch(patch)
            print(bmaj, bmin)

    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return None

    return None

def plotter_images_new(target_name, folder_images, folder_fig,  \
     max_std_lw=5, max_std_up=5, vlim = None, dx_image = 0.006,  xlim_image = 1, titles = [""],  cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_images: folders for images (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        titles: array for titles
        cmap: type of color map
    Return:
        None
    """
    ny_image = len(folder_images)
    nx_image = 1

    ## Plot
    fig, axs = plt.subplots(nx_image, ny_image, figsize=(10 * ny_image, 10 * nx_image))
    for num_now in range(np.max([nx_image,ny_image])):
        ax_now = axs[num_now]
        divider1 = make_axes_locatable(ax_now)  
        res = get_image(target_name, folder_images[num_now])
        nx,_  = np.shape(res)
        interp_ext = dx_image * nx * 0.5
        extent_for_deprojected=[-interp_ext,+interp_ext,interp_ext,-interp_ext]
        std = np.std(res)

        bmaj, bmin, bpa, beam_area = compute_beamarea_from_fits(os.path.join(folder_images[num_now],"image_%s.fits" % target_name ))

        ax_now.set_xlim(xlim_image, -xlim_image)
        ax_now.set_ylim(-xlim_image, xlim_image)
        if num_now ==0:
            ax_now.set_xlabel("$\Delta$ x [arcsec]")
            ax_now.set_ylabel("$\Delta$ y [arcsec]")
        else:
            ax_now.set_xticks([])
            ax_now.set_yticks([])            
        ax_now.set_title(titles[num_now])

        if vlim is None:
            z1 = ax_now.imshow(res, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )
            if  num_now== ny_image-1:
                cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
                cbar = plt.colorbar(z1,cax=cax1, ticks=[-max_std_lw * std , 0, max_std_up* std], label = "residual S/N")
                cbar.ax.set_yticklabels(["-%d" % max_std_lw, '0', "%d" % max_std_up])    
        else:
            z1 = ax_now.imshow(res, vmin = vlim[0], vmax = vlim[1], extent = extent_for_deprojected, cmap =  cmap )
            if  num_now== ny_image-1:
                cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
                cbar = plt.colorbar(z1,cax=cax1, label="Intensity [Jy/beam]")

        patch = mpatches.Ellipse([0.8 * xlim_image, -0.8 * xlim_image], float(bmaj), float(bmin), 90-float(bpa), fc='none', ls='solid', ec='k', lw=2.) 
        ax_now.add_patch(patch)
        print(bmaj, bmin)

    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return None

def plotter_for_res_single(target_name, res, folder_fig,  \
     max_std_lw=5, max_std_up=5,  dx_image = 0.006,  xlim_image = 1, title="",  cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_image: image
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        titles: array for titles
        cmap: type of color map
    Return:
        None
    """

    fig, ax_now = plt.subplots(figsize = (10,10) )
    divider1 = make_axes_locatable(ax_now)  
    nx,_  = np.shape(res)
    interp_ext = dx_image * nx * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    std = np.std(res)
    ax_now.set_xlim(xlim_image, -xlim_image)
    ax_now.set_ylim(-xlim_image, xlim_image)
    ax_now.set_xlabel("$\Delta x$ [arcsec]")
    ax_now.set_ylabel("$\Delta y$ [arcsec]")
    ax_now.set_title(title)
    z1 = ax_now.imshow(res, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )
    cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
    cbar = plt.colorbar(z1,cax=cax1, ticks=[-max_std_lw * std , 0, max_std_up* std], label = "residual S/N")
    cbar.ax.set_yticklabels(["-%d$\sigma$" % max_std_lw, '0', "%d$\sigma$" % max_std_up])           
    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return None


def plotter_for_image_single(target_name, res, folder_fig,  \
     vlims = None, dx_image = 0.006,  xlim_image = 1, title="",  cmap = "jet"):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_image: image
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        titles: array for titles
        cmap: type of color map
    Return:
        None
    """

    fig, ax_now = plt.subplots(figsize = (10,10) )
    divider1 = make_axes_locatable(ax_now)  
    nx,_  = np.shape(res)
    interp_ext = dx_image * nx * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    std = np.std(res)
    ax_now.set_xlim(xlim_image, -xlim_image)
    ax_now.set_ylim(-xlim_image, xlim_image)
    ax_now.set_xlabel("$\Delta x$ [arcsec]")
    ax_now.set_ylabel("$\Delta y$ [arcsec]")
    ax_now.set_title(title)
    if vlims is None:
        z1 = ax_now.imshow(res, extent = extent_for_deprojected, cmap =  cmap )
        cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
        cbar = plt.colorbar(z1,cax=cax1, label = "Flux [Jy/beam]")
    else:
        z1 = ax_now.imshow(res, vmin =vlims[0] , vmax = vlims[1], extent = extent_for_deprojected, cmap =  cmap )
        cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
        cbar = plt.colorbar(z1,cax=cax1, label = "Flux [Jy/beam]")
        
    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
    bbox_inches='tight')    
    plt.show()
    return None


def plotter_image(target_name, image_raw, folder_fig,  color_label = "$\log F$ [Jy/beam]",  \
      dx_image = 0.006,  xlim_image = 1,  cmap = "jet", vmin = 1e-5, vmax = 1e-1, title = "None"):
    
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        image_raw: background image for residual. Raw flux, log(raw flux), or sqrt(raw flux).
        folder_ress: folder for residual
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        cmap: type of color map
        vmin: min limit for plot of image_raw
        vmax: max limit for plot of image_raw
    Return:
        None
    """

    ## Preparation
    nx, ny = np.shape(image_raw)
    interp_ext = dx_image * nx * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]

    ## Plot
    fig, ax_now = plt.subplots(figsize = (10,10) )
    divider1 = make_axes_locatable(ax_now)  
    ax_now.set_xlim(xlim_image, -xlim_image)
    ax_now.set_ylim(-xlim_image, xlim_image)
    z1 = ax_now.imshow(image_raw,  vmin = vmin, vmax = vmax, extent = extent_for_deprojected, cmap =  cmap )
    cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
    plt.colorbar(z1,cax=cax1, label = "$\log F$ [Jy/beam]")
    ax_now.set_xlabel("$\Delta$ ra [arcsec]", fontsize =32)
    ax_now.set_ylabel("$\Delta$ dec [arcsec]", fontsize =32)
    if title is None:
        ax_now.set_title(target_name, fontsize=32)
    else:
        ax_now.set_title(title, fontsize=32)
    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()

    return None

def plotter_image_with_zoom_view(target_name, res, folder_fig, rad_circle_dark_arr, rad_circle_bright_arr,r_arr_angle , angle_arr,  \
      max_std_lw, max_std_up,  dx_image = 0.006,  xlim_image = 1,  cen_circle =0, cmap = "jet",  title = "None", out_name = None ,line =True):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        image_raw: background image for residual. Raw flux, log(raw flux), or sqrt(raw flux).
        folder_ress: folder for residual
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        cmap: type of color map
        vmin: min limit for plot of image_raw
        vmax: max limit for plot of image_raw
    Return:
        None
    """

    ## Preparation
    nx, ny = np.shape(res)
    interp_ext = dx_image * nx * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]

    std = np.std(res[np.isfinite(res)])
    ## Plot
    fig, ax_now = plt.subplots(figsize = (10,10) )
    lim_polar = xlim_image# * 1.15
    divider1 = make_axes_locatable(ax_now)  
    ax_now.set_xlim(lim_polar , -lim_polar )
    ax_now.set_ylim(-lim_polar , lim_polar )
    z1 = ax_now.imshow(res,  vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )

    axins = ax_now.inset_axes([0.7, 0.7, 0.29, 0.29])
    axins.imshow(res,  vmin = -max_std_lw * std , vmax = max_std_up* std,  extent=extent_for_deprojected, origin="lower", cmap =  cmap )
    # sub region of the original image
    x1, x2, y1, y2 = 0.28, -0.28, 0.28,  -0.28
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])


    ## 

    cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
    cbar = plt.colorbar(z1, cax=cax1, ticks=[-max_std_lw * std , 0, max_std_up* std], label = "residual S/N")
    cbar.ax.set_yticklabels(["-%d" % max_std_lw, '0', "%d" % max_std_up]) 
    ax_now.set_xlabel("$\Delta x$ [arcsec]", fontsize =32)
    ax_now.set_ylabel("$\Delta y$ [arcsec]", fontsize =32)
    if title is None:
        ax_now.set_title(target_name, fontsize=32)
    else:
        ax_now.set_title(title, fontsize=32)
    if out_name is not None:
        plt.savefig(os.path.join(folder_fig, "%s_%s.pdf" % (target_name,  out_name)), 
                bbox_inches='tight')  
    else:
        plt.savefig(os.path.join(folder_fig, "%s.pdf" % (target_name)), 
            bbox_inches='tight')      
    plt.show()

    return None


def plotter_image_spiral(target_name, res, folder_fig, rad_circle_dark_arr, rad_circle_bright_arr,r_arr_angle , angle_arr,  \
      max_std_lw, max_std_up,  dx_image = 0.006,  xlim_image = 1,  cen_circle =0, cmap = "jet",  title = "None", out_name = None ,line =True):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        image_raw: background image for residual. Raw flux, log(raw flux), or sqrt(raw flux).
        folder_ress: folder for residual
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        cmap: type of color map
        vmin: min limit for plot of image_raw
        vmax: max limit for plot of image_raw
    Return:
        None
    """

    ## Preparation
    nx, ny = np.shape(res)
    interp_ext = dx_image * nx * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]


    std = np.std(res[np.isfinite(res)])
    ## Plot
    fig, ax_now = plt.subplots(figsize = (10,10) )
    lim_polar = xlim_image# * 1.15
    divider1 = make_axes_locatable(ax_now)  
    ax_now.set_xlim(lim_polar , -lim_polar )
    ax_now.set_ylim(-lim_polar , lim_polar )
    z1 = ax_now.imshow(res,  vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )

    if rad_circle_dark_arr is not None:
        circle_arr = make_cirle(rad_circle_dark_arr , cen_circle, color="g", linestyle ="dashed", lw = 4, alpha=0.6)
        for circle in circle_arr:
            ax_now.add_patch(circle)

    if rad_circle_bright_arr is not None:
        circle_arr = make_cirle(rad_circle_bright_arr , cen_circle, color="k", linestyle ="dashed", lw =4, alpha=0.6)
        for circle in circle_arr:
            ax_now.add_patch(circle)

    if r_arr_angle is not None:
        x = r_arr_angle * np.cos(angle_arr)
        y = r_arr_angle * np.sin(angle_arr)
        color_sc = "k"
        if line:
            ax_now.plot(x[r_arr_angle<lim_polar], y[r_arr_angle<lim_polar], color =color_sc, lw=2, alpha = 0.5)
            ax_now.plot(-x[r_arr_angle<lim_polar], -y[r_arr_angle<lim_polar], color =color_sc, lw=2, alpha = 0.5)
            ax_now.scatter(x[r_arr_angle<lim_polar], y[r_arr_angle<lim_polar], color =color_sc, alpha = 0.5)
            ax_now.scatter(-x[r_arr_angle<lim_polar], -y[r_arr_angle<lim_polar], color =color_sc, alpha = 0.5)
        else:
            ax_now.scatter(x[r_arr_angle<lim_polar], y[r_arr_angle<lim_polar], color =color_sc, s = 50, alpha = 0.5)
            ax_now.scatter(-x[r_arr_angle<lim_polar], -y[r_arr_angle<lim_polar], color =color_sc,s = 50,  alpha = 0.5)            
    cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
    cbar = plt.colorbar(z1, cax=cax1, ticks=[-max_std_lw * std , 0, max_std_up* std], label = "residual S/N")
    cbar.ax.set_yticklabels(["-%d" % max_std_lw, '0', "%d" % max_std_up]) 
    ax_now.set_xlabel("$\Delta x$ [arcsec]", fontsize =32)
    ax_now.set_ylabel("$\Delta y$ [arcsec]", fontsize =32)
    if title is None:
        ax_now.set_title(target_name, fontsize=32)
    else:
        ax_now.set_title(title, fontsize=32)
    if out_name is not None:
        plt.savefig(os.path.join(folder_fig, "%s_%s.pdf" % (target_name,  out_name)), 
                bbox_inches='tight')  
    else:
        plt.savefig(os.path.join(folder_fig, "%s.pdf" % (target_name)), 
            bbox_inches='tight')      
    plt.show()

    return None


def plotter_image_res_cont_new(target_name, image_raw, folder_res, folder_fig,  \
     max_std_lw=5, max_std_up=5,  dx_image = 0.006,  xlim_image = 1,  cmap = "jet", vmin = 1e-5, vmax = 1e-1, title = None):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        image_raw: background image for residual. Raw flux, log(raw flux), or sqrt(raw flux).
        folder_ress: folder for residual
        folder_fig: folder for output fig
        max_std_lw, max_std_up: lower, upper bounds for plots 
        dx_image: pixel scale for image
        xlim_image: x-range for image
        cmap: type of color map
        vmin: min limit for plot of image_raw
        vmax: max limit for plot of image_raw
    Return:
        None
    """

    ## Preparation
    res = get_image(target_name, folder_res)
    nx, ny = np.shape(res)
    interp_ext = dx_image * nx * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]

    ## Plot
    fig, ax_now = plt.subplots(figsize = (10,10) )
    divider1 = make_axes_locatable(ax_now)  
    std = np.std(res[np.isfinite(res)])
    ax_now.set_xlim(xlim_image, -xlim_image)
    ax_now.set_ylim(-xlim_image, xlim_image)
    z1 = ax_now.imshow(image_raw,  vmin = vmin, vmax = vmax, extent = extent_for_deprojected, cmap =  cmap )
    z2 =ax_now.contour(np.flip(res, axis=0), levels = [-std * max_std_lw , std * max_std_up], colors=["cyan","w"], extent = extent_for_deprojected, alpha = 1 )
    cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
    plt.colorbar(z1,cax=cax1, label = "$\log F$ [Jy/beam]")
    ax_now.set_xlabel("$\Delta x$ [arcsec]", fontsize =32)
    ax_now.set_ylabel("$\Delta y$ [arcsec]", fontsize =32)
    if title is None:
        ax_now.set_title(target_name, fontsize=32)
    else:
        ax_now.set_title(title, fontsize=32)
    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()

    return None
        
def plotter_image_res_cont(target_name, folder_raw, folder_res, folder_mcmc, folder_fig, pickle_name_before_deprojected, pickle_name_deprojected, \
     pickle_name_raw_before_deprojected, pickle_name_raw_deprojected, max_std_lw, max_std_up, max_flux_lw = None, max_flux_up = None, pix_interp = 500, dx_interp = 0.01, rad_circle_arr = [0.25, 0.50, 1, 1.5], cen_circle =0, \
        dx_image = 0.006, cen = 0, plot_pix_image_scale = 1, plot_pix_image_scale_for_zoom = 0.5, nx_image = 2, ny_image = 2, cmap = "jet", vmin = -6, vmax = -2, log_scale = True):
    """
    Plotting image, residual, deprojected residual, & zoom deprojected residual. 

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )
        folder_mcmc: folder containing mcmc result
        folder_fig: folder for output fig
        pickle_name_before_deprojected: pickle file for (residual) image
        pickle_name_deprojected: pickle file  for deprojected (residual) image
        max_std_lw, max_std_up: lower, upper bounds for plots 
        pix_interp: number of pixels for interpolated deprojected image (pix_interp * pix_interp)
        dx_interp: angular size of one pixel for interpolated deprojected image
        rad_circle_arr: radii for circles plotted for reference
        cen_circle: center of circle
        dx_image: pixel scale for image
        cen: central coordinate of image
        plot_pix_image_scale: scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
        plot_pix_image_scale_for_zoom: zooming scale parameter determines the half width of plotted image in unit of outer radius of disk (r_max_interp)
    Return:
        None
    """

    ## Preparation
    image = get_image(target_name, folder_raw)
    res = get_image(target_name, folder_res)

    if image is None:
        return None, None
    r_max_interp, interp_f = make_radial_profile_target(target_name, folder_raw, folder_mcmc, dx_image)
    plot_pix_image = plot_pix_image_scale * r_max_interp
    plot_pix_image_zoom =plot_pix_image_scale_for_zoom  * r_max_interp
    interp_ext = dx_interp * pix_interp * 0.5
    extent_for_deprojected=[interp_ext,-interp_ext,interp_ext,-interp_ext]
    extent_for_deprojected_cont=[interp_ext,-interp_ext,-interp_ext,+interp_ext]
    nx, ny = np.shape(image)
    d_extent = dx_image * (nx/2)
    extent=[d_extent,-d_extent,+d_extent,-d_extent]

    ## Plot
    fig, ax_now = plt.subplots(figsize = (10,10) )
    
    divider1 = make_axes_locatable(ax_now)  
    interpolator, image_out = make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected, \
        pickle_name_deprojected, res, dx_image, pix_interp, dx_interp)
    interpolator_raw, image_out_raw = make_deprojected_image(target_name,folder_mcmc, pickle_name_raw_before_deprojected, \
        pickle_name_raw_deprojected, image, dx_image, pix_interp, dx_interp)
    std = np.std(image_out[np.isfinite(image_out)])
    print(std)
    circle_arr = make_cirle(rad_circle_arr , cen_circle)
    ax_now.set_xlim(cen_circle+plot_pix_image, cen_circle-plot_pix_image)
    ax_now.set_ylim(cen_circle-plot_pix_image, cen_circle+plot_pix_image)
    if max_flux_lw is None:
        z2 =ax_now.contour(image_out, levels = [-std * max_std_lw , std * max_std_up], colors=["g","k"], extent = extent_for_deprojected_cont, alpha = 1 )
    else:
        z2 =ax_now.contour(image_out, levels = [max_flux_lw , max_flux_up], colors=["g","k"], extent = extent_for_deprojected_cont, alpha = 1)
    if log_scale:
        z1 = ax_now.imshow(np.log10(image_out_raw),  vmin = vmin, vmax = vmax, extent = extent_for_deprojected, cmap =  cmap )
    else:
        z1 = ax_now.imshow(image_out_raw,  vmin = vmin, vmax = vmax, extent = extent_for_deprojected, cmap =  cmap )
    #z1 = ax_now.imshow(image_out_raw,  extent = extent_for_deprojected, cmap =  cmap )
    cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
    plt.colorbar(z1,cax=cax1, label = "$\log F$ [Jy/beam]")

    ax_now.set_xlabel("$\Delta x$ [arcsec]")
    ax_now.set_ylabel("$\Delta y$ [arcsec]")
    ax_now.set_title(target_name)
    plt.savefig(os.path.join(folder_fig, "%s.pdf" % target_name), 
               bbox_inches='tight')    
    plt.show()
    return image_out_raw, image_out
    