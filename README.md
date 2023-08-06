# georadial
Estimating itensity profile & geometry & hyperparameters proto-planetary disks in ALMA 

## Install 
For normal install, 
* python setup.py install
* *For conda/pip env*, go to parental directory of georadial, and "pip install ./georadial"

For developers, 
* python setup.py develop
*  *For conda/pip env*, go to parental directory of georadial, and "pip install -e ./georadial"

## MCMC run
1. Download test data from https://github.com/2ndmk2/dsharp_averaged_data
2. go to "tests" folder
3. put test data in a data directory and do "run.sh"
   ** This step is most time consuming. Need ~1.5 hours to finish 32 waklers * 1000 steps with 16 cores.

## Postprocess after MCMC
1. Calculate model itensity & visibility profile using "model_calc.py"

## Plot
1. 
