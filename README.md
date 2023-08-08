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
4. Run "python run_sampling.py --config ./paras/mcmc_config.dat --initial_para ./paras/AS209_paradic.dat --prior ./paras/prior.dat --visfile ./vis_data/AS209_continuum_averaged.vis.npz"  
   **This step is most time consuming. Need ~1.5 hours to finish (32 walkers, 1000 steps) with 16 cores.**

### 2) Postprocess after MCMC
1. Calculate model itensity & visibility profile using "model_calc.py"

### 3) See result
1. Go to ipynb folder, and use mcmc_plotter. Modify "samplefile" & "modelfile" to make them correct paths. 
