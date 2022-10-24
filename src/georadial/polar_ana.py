
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


def interpolate_image_ND(image, dx = 0.01, dy= 0.01):
    ## 
    nx, ny = np.shape(image)
    x = make_coordinate(nx, dx)
    y = make_coordinate(ny, dy)
    xx, yy = np.meshgrid(x, y)
    xx = -xx
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


def main_interpolated_image(pickle_name, pickle_name2, res, dx_original_image, x_coord, cosi, pa):

    if not os.path.exists(pickle_name):
        int_f = interpolate_image_ND(res, dx_original_image, dx_original_image)
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


def get_raw_and_res(target_name, folder_raw, folder_res):
    """
    Obtain image & residual from folderes

    Params:
        target_name: target_name for target
        folder_raw: folder for image (os.path.join(folder_res,"image_%s.fits" % target_name )
        folder_res: folder for residual (os.path.join(folder_raw,"image_%s.fits" % target_name )

    Return:
        image_raw: 2d array for image 
        image_subtracted: 2d array for residual
    """

    raw_file = os.path.join(folder_raw,"image_%s.fits" % target_name )
    subtracted_file = os.path.join(folder_res,"image_%s.fits" % target_name )
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


def make_deprojected_image(target_name,folder_mcmc,pickle_name_before_deprojected, pickle_name_deprojected,  res, dx_image, pix_interp, dx_interp ):
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
    image, res = get_raw_and_res(target_name, folder_raw, folder_res)
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
    image, res = get_raw_and_res(target_name, folder_raw, folder_res)
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
    image, res = get_raw_and_res(target_name, folder_raw, folder_res)
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
    image, res = get_raw_and_res(target_name, folder_raw, folder_res)
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
    image, res = get_raw_and_res(target_name, folder_raw, folder_res)
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
        #ax_now.scatter(cen_circle, cen_circle, s = 10, color="r")
        #for circle_now in circle_arr:
        #    ax_now.add_patch(circle_now)
        #z2 =ax_now.contour(image_out, levels = [-2 * std , 2 * std], colors=["w","k"], extent = extent_for_deprojected_cont  )
        z1 = ax_now.imshow(image_out, vmin = -max_std_lw * std , vmax = max_std_up* std, extent = extent_for_deprojected, cmap =  cmap )
        cax1 = divider1.append_axes("right", size="5%", pad=0.1) 
        cbar = plt.colorbar(z1,cax=cax1, ticks=[-max_std_lw * std , 0, max_std_up* std], label = "residual S/N")
        cbar.ax.set_yticklabels(["-%d" % max_std_lw, '0', "%d" % max_std_up])           

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
    image, res = get_raw_and_res(target_name, folder_raw, folder_res)
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
    