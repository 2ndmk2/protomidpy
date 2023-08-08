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
1. Go to ipynb folder, and use mcmc_plotter.  
   Modify "samplefile" & "modelfile" to make them correct paths.

------

### Config files
**[mcmc_config.dat]**  
Parameters for emcee and model  

*Nrad*: Number of radial point for model intensity)  
*Nbin*: Determine binning grid. Grid size is (2*Nbin+1, 2*Nbin+1)  
*Dpix*: Radial spacing for model [arcsec]. Outer disk radius is determined as "Rout = Nrad * Dpix"  
*Nwalker*: Number of walkers for emcee)  
*Nchain*: Number of chains for emcee)  
*cov*: Choice of regularization (default is RBF)  
*out_folder*: Path to output folder  

**[AS209_paradic.dat]**  
Parameters determining initial positions for mcmc.  
They are randonly generated with uniform distribution [value-scatter/2,value+scatter/2].  

**[prior.dat]**  
Parameters determining parameter priors.  
prior for alpha: Uniform prior for [log10_alpha_min, log10_alpha_max]  
prior for gamma: Uniform prior for [min_scale [arcsec], max_scale  [arcsec]]  
prior for disk center: Uniform prior for [-delta_pos [arcsec], delta_pos  [arcsec]]  
