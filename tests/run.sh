python -m unittest tests/test_warp.py

PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python tests/run_sampling.py \
  --n_process 2 \
  --config ./tests/paras/warp_smoke_config.dat \
  --initial_para ./tests/paras/warp_smoke_paradic.dat \
  --prior ./tests/paras/warp_smoke_prior.dat \
  --visfile ./tests/AS209_continuum_averaged.vis.npz

PYTHONPATH=src python tests/model_calc.py \
  --n_sample_for_rad 8 \
  --n_burnin 0 \
  --visfile ./tests/AS209_continuum_averaged.vis.npz \
  --mcmc_result_file ./tests/result_warp_smoke/AS209_continuum_averaged.vis_mcmc.npz \
  --out_file_for_model ./tests/result_warp_smoke/AS209_continuum_averaged_model.npz
