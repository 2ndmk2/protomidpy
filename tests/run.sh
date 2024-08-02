python run_sampling.py --n_process 4 --config ./paras/mcmc_config.dat  --initial_para ./paras/AS209_paradic.dat --prior ./paras/prior.dat --visfile ./vis_data/AS209_continuum_averaged.vis.npz


python model_calc.py --n_sample_for_rad 20 --n_burnin 20000 --visfile ./vis_data/AS209_continuum_averaged.vis.npz --mcmc_result_file ./result/AS209_continuum_averaged.vis_mcmc.npz --initial_para ./paras/AS209_paradic.dat --prior ./paras/prior.dat --out_file_for_model ./result/AS209_continuum_averagedmodel.npz \