
NCORE=8

python3 run_sampling_warp.py  --n_process $NCORE  --visfile ./AS209_continuum_averaged.vis.npz --prior ./paras/warp_prior.dat --initial_para ./paras/warp_paradic.dat  --config ./paras/mcmc_config.dat
