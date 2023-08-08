# georadial
Estimating itensity profile & geometry & hyperparameters proto-planetary disks in ALMA 

## Install 
For normal install, 
* python setup.py install or pip install ./georadial

For developers, 
* python setup.py develop or pip install -e ./georadial
    
## Prerequisites
- astropy, emcee, corner, matplotlib, numpy, scipy, pandas, (Jupyter notebook)

## Usage

### 1) MCMC run
1. Download test data from https://github.com/2ndmk2/dsharp_averaged_data
2. Make "vis_data" folder ant put the data there
3. Move to "tests" folder
4. Run "bash run.sh"  
   **This step is most time consuming. Need ~1.5 hours to finish (32 walkers, 1000 steps) with 16 cores.**

### 2) Postprocess after MCMC
1. Calculate model itensity & visibility profile using "model_calc.py"

### 3) See result
1. Go to ipynb folder, and use mcmc_plotter. Modify "samplefile" & "modelfile" to make them correct paths.

### Config files
**[mcmc_config.dat]**  

Nrad 200 (Number of radial point for model intensity)  
Nbin 500 (Determine binning grid. Grid size is (2*Nbin+1, 2*Nbin+1)  
Dpix 0.01 (Radial spacing for model [arcsec]. Rout = Nrad * Dpix) 
Nwalker 32  (Number of walkers for emcee)  
Nchain 1000  (Number of chains for emcee)  
cov RBF  (Choice of regularization)
out_folder ./result (Output folder)

**[AS209_paradic.dat]**
Parameters determining initial positions for mcmc.  
They are randonly generated with uniform distribution [value - 0.5*scatter, value+0.5*scatter].

gamma_value 0.03  
log10_alpha_value -1  
gamma_scatter 0.01  
log10_alpha_scatter 0.01  
pa_value 85.7600  
cosi_value 0.8195  
pa_scatter 0.01  
cosi_scatter 0.01  
delta_pos_x 0.0019  
delta_pos_y -0.0025  
delta_pos_scatter 0.05  

**[prior.dat]**
Parameters determining parameter priors. 
Alpha: Uniform prior for [log10_alpha_min, log10_alpha_max]  
gamma: Uniform prior for [min_scale, max_scale ]  
Disk center: Uniform prior for [-delta_pos, delta_pos ]  

log10_alpha_min -4  
log10_alpha_max 5  
min_scale 0.01  
max_scale 0.15  
delta_pos 1.0  

