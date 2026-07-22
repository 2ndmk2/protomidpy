import shutil
import numpy as np
import os
import sys
import glob


def load_ms(oms_name, oms_path="."):

	# Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
	# note that "uvw" is unit of "m"
	tb.open(oms_path+'/'+oms_name+'.ms')
	data   = tb.getcol("DATA")
	flag   = tb.getcol("FLAG")
	uvw    = tb.getcol("UVW") #m
	weight = tb.getcol("WEIGHT")
	spwid  = tb.getcol("DATA_DESC_ID")
	sigma=tb.getcol('SIGMA')  
	tb.close()


	# Get the frequency information
	tb.open(oms_path+'/'+oms_name+'.ms/SPECTRAL_WINDOW')
	freq = tb.getcol("CHAN_FREQ")[0]/1e9 ##GHz
	freq_pol = tb.getcol("CHAN_FREQ")/1e9 ##GHz
	c_const = 299792458.0 * 1e3/1e9 #mm Ghz
	lambda_arr = c_const/freq
	tb.close()



	# Get rid of any flagged columns 
	flagged   = np.all(flag, axis=(0, 1))
	unflagged = np.squeeze(np.where(flagged == False))
	data   = data[:,:,unflagged]
	weight = weight[:,unflagged]
	uvw    = uvw[:,unflagged]
	spwid    = spwid[unflagged]
	sigma = sigma[:,unflagged]

	# Assign uniform spectral-dependence to the weights (pending CASA improvements)
	sp_wgt = np.zeros_like(data.real)
	print(np.shape(sp_wgt))
	for i in range(len(freq_pol)): sp_wgt[:,i,:] = weight

	# (weighted) average the polarizations
	Re  = np.sum(data.real*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
	Im  = - np.sum(data.imag*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
	Vis = np.squeeze(Re + 1j*Im)
	Wgt = np.squeeze(np.sum(sp_wgt, axis=0))

	u_freq = []
	v_freq =[]
	vis_freq=[]
	wgt_freq =[]
	nfreq = len(freq)
	freq_arr = []
	for i_freq in range(nfreq):
		flag_spw = spwid==i_freq
		u_freq.append(uvw[0,:][flag_spw] * 1e3/lambda_arr[i_freq])
		v_freq.append(uvw[1,:][flag_spw] * 1e3/lambda_arr[i_freq])
		vis_freq.append(Vis[flag_spw])
		wgt_freq.append(Wgt[flag_spw])
		freq_arr.append(np.ones(len(spwid[flag_spw])) * freq [i_freq])

	# output to numpy file
	os.system('rm -rf '+oms_name+'.vis.npz')

	##
	u_obs_ravel = np.concatenate(u_freq)
	v_obs_ravel = np.concatenate(v_freq)
	vis_obs_ravel = np.concatenate(vis_freq)
	freq_obs_ravel = np.concatenate(freq_arr)
	wgt_obs_ravel = np.concatenate(wgt_freq)
	np.savez(oms_name+'.vis', u_obs = u_obs_ravel, v_obs = v_obs_ravel, vis_obs = vis_obs_ravel, freq_obs= freq_obs_ravel, wgt_obs = wgt_obs_ravel)#, v_freq=v_freq, vis_freq=vis_freq, wgt_freq=wgt_freq, freq_arr=freq, lambda_arr= lambda_arr, sigma=sigma)
	return freq

if __name__ == '__main__':

	## Main
	folder="./measurement_folder" ## folder containing ms files
	files = glob.glob(folder + "/*.ms")

	## convert each ms to npz file
	for file in files:
		file_last = file.split("/")[-1]
		load_ms(file_last.replace(".ms",""), folder)
