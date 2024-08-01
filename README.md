# georadial
Estimating itensity radial profile & geometry & hyperparameters proto-planetary disks in ALMA  
- For technical details of algorithm and mathematic background, please read Aizawa, Muto, Momose 2024  
https://academic.oup.com/mnras/article/532/2/1361/7699107

## Install 
For normal install, 
* python setup.py install or pip install ./georadial

For developers, 
* python setup.py develop or pip install -e ./georadial
    
## Prerequisites
- astropy, emcee, corner, matplotlib, numpy, scipy, pandas, (Jupyter notebook)


## Config files
**[mcmc_config.dat]**  
Parameters for emcee and model  

- *Nrad*: Number of radial point for model intensity)  
- *Nbin*: Determine binning grid. Grid size is (2*Nbin+1, 2*Nbin+1)  
- *Dpix*: Radial spacing for model [arcsec]. Outer disk radius is determined as "Rout = Nrad * Dpix"  
- *Nwalker*: Number of walkers for emcee
- *Nchain*: Number of chains for emcee 
- *out_folder*: Path to output folder  

**[AS209_paradic.dat]**  
Parameters determining initial positions for mcmc.  
They are randonly generated with uniform distribution [value-scatter/2,value+scatter/2].  

**[prior.dat]**  
Parameters determining parameter priors.  
- prior for alpha: Uniform prior for [log10_alpha_min, log10_alpha_max]  
- prior for gamma: Uniform prior for [min_scale [arcsec], max_scale  [arcsec]]  
- prior for disk center: Uniform prior for [-delta_pos [arcsec], delta_pos  [arcsec]]  

## Format of Input Visibility data
Download test data from https://github.com/2ndmk2/dsharp_averaged_data
   - ** The data, should contain following items  in form of ".npz" file 
       - "u_obs": Spatial frequency "u" [lambda]
       - "v_obs": Spatial frequency "v" [lambda]
       - "vis_obs": Visibility 
       - "wgt_obs": Weights
       

## Run
### 1) MCMC run 
Run "run_sampling.py" in tests folder by selecting proper config files as options. "n_process" is the number of cores to be used in emcee. 
- python run_sampling.py --n_process 4 --config ./paras/mcmc_config.dat --initial_para ./paras/AS209_paradic.dat --prior ./paras/prior.dat --visfile ./vis_data/AS209_continuum_averaged.vis.npz


### 2) Postprocess Calculation
Run "model_calc.py" in tests folder by selecting proper config files as options. "n_process" is the number of cores to be used in emcee. 

1. Modify "model_calc.py" as you like
    - *n_sample_for_rad*: Number of smaples for intensity profiles
    - *n_burnin*: Number of Burnin samples
    - *mcmc_result_file*: path to "~~vis_mcmc.npz" output from run.sh
    - *visfile*: path to "~~vis.npz" output from run.sh
    - *out_file_for_model*: path to output file from "model_calc.py"
2. Calculate model itensity & visibility profile using "model_calc.py"

### 3) See result
1. Run mcmc_plotter.ipynb
   - *samplefile*: path to "~~vis_mcmc.npz" output from run.sh
   - *modelfile*: path to output file from model_calc.py
   
------

