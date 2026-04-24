import argparse
import os
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protomidpy import data_gridding
from protomidpy import hankel
from protomidpy import mcmc_utils
from protomidpy import sample
from protomidpy import utils

ARCSEC_TO_RAD = 1 / 206265.0


def build_model_products(mcmc_result_file, visfile, out_file_for_model, n_sample_for_rad=20, n_burnin=20000):
    mcmc_result = np.load(mcmc_result_file)
    sample_chain = mcmc_result["sample"]
    log_posterior = mcmc_result["log_prior"] + mcmc_result["log_likelihood"]

    n_total = len(sample_chain)
    burnin = min(n_burnin, max(n_total - 1, 0))
    sample_goods = sample_chain[burnin:, :]
    if len(sample_goods) == 0:
        sample_goods = sample_chain

    sample_best = sample_chain[np.argmax(np.ravel(log_posterior)), :]
    n_bin_log = int(mcmc_result["n_bin_log"])
    nrad = int(mcmc_result["nrad"])
    dpix = float(mcmc_result["dpix"]) * ARCSEC_TO_RAD
    cov = str(mcmc_result["cov"])
    r_out = nrad * dpix
    q_min_max_bin = [float(mcmc_result["qmin"]), float(mcmc_result["qmax"])]

    u_d, v_d, vis_d, wgt_d, freq_d = utils.load_obsdata(visfile)
    data_d = np.append(vis_d.real, vis_d.imag)
    coord_for_grid_lg, rep_positions_for_grid_lg, uu_for_grid_pos_lg, vv_for_grid_pos_lg = data_gridding.log_gridding_2d(
        q_min_max_bin[0], q_min_max_bin[1], n_bin_log
    )
    u_grid_2d, v_grid_2d, vis_grid_2d, noise_grid_2d, sigma_mat_2d, d_data, binnumber = data_gridding.data_binning_2d(
        u_d, v_d, vis_d, wgt_d, coord_for_grid_lg
    )
    r_n, jn, qmax, q_n, h_mat_model, q_dist_2d_model, n_d, r_dist, d_a_minus1_d, logdet_for_sigma_d = hankel.prepare(
        r_out, nrad, d_data, sigma_mat_2d
    )

    flux_arr = []
    n_take = min(n_sample_for_rad, len(sample_goods))
    sample_random_selected = sample_goods[np.random.choice(np.arange(len(sample_goods)), n_take, replace=False)]
    for sample_now in sample_random_selected:
        flux_sampled, h_mat = sample.sample_radial_profile(
            r_dist, sample_now, u_grid_2d, v_grid_2d, r_out, nrad, dpix, d_data, sigma_mat_2d, q_dist_2d_model, h_mat_model, cov=cov
        )
        flux_arr.append(flux_sampled)

    sample_one_taken, h_mat = sample.sample_radial_profile(
        r_dist, sample_best, u_grid_2d, v_grid_2d, r_out, nrad, dpix, d_data, sigma_mat_2d, q_dist_2d_model, h_mat_model, cov=cov
    )
    h_mat, q_dist, d_real_mod, d_imag_mod, vis_model_real, vis_model_imag, u_mod, v_mod = mcmc_utils.obs_model_comparison(
        sample_one_taken, u_d, v_d, sample_best, data_d, r_out, nrad, dpix
    )
    vis_model, residual = mcmc_utils.make_model_and_residual(u_d, v_d, sample_best, sample_one_taken, vis_d, r_out, nrad, dpix)

    out_dir = os.path.dirname(out_file_for_model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        out_file_for_model,
        r_n=r_n,
        param_map=sample_best,
        params_random_selected=sample_random_selected,
        flux_map_sample=sample_one_taken,
        flux_random_samples=flux_arr,
        vis_model_undeprojected=vis_model,
        residual_undeprojected=residual,
        qdist_deprojected=q_dist,
        vis_model_deprojected=vis_model_real + 1j * vis_model_imag,
        data_deprojected=d_real_mod + 1j * d_imag_mod,
        data_weights=wgt_d,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample_for_rad", type=int, default=20)
    parser.add_argument("--n_burnin", type=int, default=20000)
    parser.add_argument("--mcmc_result_file", default=str(TESTS_DIR / "result" / "AS209_continuum_averaged.vis_mcmc.npz"))
    parser.add_argument("--visfile", default=str(TESTS_DIR / "AS209_continuum_averaged.vis.npz"))
    parser.add_argument("--out_file_for_model", default=str(TESTS_DIR / "result" / "AS209_continuum_averagedmodel.npz"))
    args = parser.parse_args()

    build_model_products(
        args.mcmc_result_file,
        args.visfile,
        args.out_file_for_model,
        n_sample_for_rad=args.n_sample_for_rad,
        n_burnin=args.n_burnin,
    )


if __name__ == "__main__":
    main()
