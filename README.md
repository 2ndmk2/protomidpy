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

- mcmc_config.dat  
Nrad 200  
Nbin 500  
Dpix 0.01  
Nwalker 32  
Nchain 1000  
cov RBF  
out_folder ./result  
  
- AS209_paradic.dat  
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

- prior.dat  
log10_alpha_min -4  
log10_alpha_max 5  
min_scale 0.01  
max_scale 0.15  
delta_pos 1.0  

