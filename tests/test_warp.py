import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


def load_protomidpy_modules():
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "protomidpy"
    package = types.ModuleType("protomidpy")
    package.__path__ = [str(root)]
    sys.modules["protomidpy"] = package

    loaded = {}
    for name in ["hankel", "covariance", "prob", "mcmc_utils", "sample"]:
        spec = importlib.util.spec_from_file_location(f"protomidpy.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"protomidpy.{name}"] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


MODULES = load_protomidpy_modules()
hankel = MODULES["hankel"]
prob = MODULES["prob"]
mcmc_utils = MODULES["mcmc_utils"]


class WarpGeometryTests(unittest.TestCase):

    def setUp(self):
        self.u = np.array([1.0e6, 2.0e6, -1.5e6])
        self.v = np.array([0.5e6, -0.2e6, 1.2e6])
        self.n_vis = len(self.u)
        self.r_out = 0.5 * hankel.ARCSEC_TO_RAD
        self.nrad = 8
        self.dpix = self.r_out / self.nrad
        self.factor_all, self.r_pos = hankel.make_hankel_matrix_kataware(self.r_out, self.nrad, self.dpix)
        self.r_n, self.jn, self.qmax, self.q_n = hankel.make_collocation_points(self.r_out, self.nrad)
        self.warp_params = {
            "cosi_outer": 0.5,
            "pa_outer": 0.7,
            "r_transition": 0.2 * hankel.ARCSEC_TO_RAD,
            "r_width": 0.05 * hankel.ARCSEC_TO_RAD,
        }

    def test_fixed_geometry_hankel_matches_manual_construction(self):
        q_dist = hankel.make_q_dist_at_inc_pa(self.u, self.v, 0.7, 0.3)
        h_manual = hankel.make_hankel_matrix_from_kataware(
            q_dist, self.factor_all, self.r_pos, self.qmax, 0.7
        )
        h_full = hankel.make_hankel_at_inc_pa_w_offset(
            self.u, self.v, 0.7, 0.3, 0.0, 0.0, self.r_out, self.nrad,
            self.factor_all, self.r_pos, self.dpix, self.qmax,
        )
        np.testing.assert_allclose(h_full[:self.n_vis], h_manual)
        np.testing.assert_allclose(h_full[self.n_vis:], 0.0)

    def test_warp_hankel_uses_columnwise_cosi_factor(self):
        q_dist, cosi_profile, _ = hankel.make_q_dist_at_inc_pa(
            self.u, self.v, 0.8, 0.2, radii=self.r_n, warp_params=self.warp_params, return_profile=True
        )
        h_manual = hankel.make_hankel_matrix_from_kataware(
            q_dist, self.factor_all, self.r_pos, self.qmax, cosi_profile
        )
        h_full = hankel.make_hankel_at_inc_pa_w_offset(
            self.u, self.v, 0.8, 0.2, 0.0, 0.0, self.r_out, self.nrad,
            self.factor_all, self.r_pos, self.dpix, self.qmax, warp_params=self.warp_params,
        )
        np.testing.assert_allclose(h_full[:self.n_vis], h_manual)
        np.testing.assert_allclose(h_full[self.n_vis:], 0.0)

    def test_warp_prior_and_initial_walkers(self):
        theta = np.array([0.03, -1.0, 0.8, 0.2, 0.0, 0.0, 0.5, 0.7, 0.2, 0.05])
        prior = {
            "min_scale": 0.01,
            "max_scale": 0.15,
            "log10_alpha_min": -4,
            "log10_alpha_max": 5,
            "delta_pos": 1.0,
            "warp_r_transition_min": 0.0,
            "warp_r_transition_max": 1.0,
            "warp_r_width_min": 1e-4,
            "warp_r_width_max": 1.0,
        }
        self.assertTrue(np.isfinite(prob.log_prior_geo(theta, prior)))

        prior_bad = dict(prior)
        prior_bad["warp_r_width_min"] = 0.1
        self.assertEqual(prob.log_prior_geo(theta, prior_bad), -np.inf)

        para_dic = {
            "gamma_value": 0.03,
            "gamma_scatter": 0.01,
            "log10_alpha_value": -1.0,
            "log10_alpha_scatter": 0.01,
            "pa_value": 85.0,
            "pa_scatter": 0.1,
            "cosi_value": 0.8,
            "cosi_scatter": 0.02,
            "delta_pos_x": 0.0,
            "delta_pos_y": 0.0,
            "delta_pos_scatter": 0.02,
            "warp_cosi_out_value": 0.5,
            "warp_cosi_out_scatter": 0.02,
            "warp_pa_out_value": 40.0,
            "warp_pa_out_scatter": 0.2,
            "warp_r_transition_value": 0.2,
            "warp_r_transition_scatter": 0.05,
            "warp_r_width_value": 0.05,
            "warp_r_width_scatter": 0.01,
        }
        walkers = mcmc_utils.make_initial_geo_offset(para_dic, 6)
        self.assertEqual(walkers.shape, (6, 10))

    def test_warp_obs_model_comparison_returns_expected_shapes(self):
        theta = np.array([0.03, -1.0, 0.8, 0.2, 0.0, 0.0, 0.5, 0.7, 0.2, 0.05])
        i_model = np.linspace(1.0, 0.1, self.nrad)
        d_data = np.concatenate([np.ones(self.n_vis), np.zeros(self.n_vis)])
        h_mat, q_eff, d_real_mod, d_imag_mod, vis_model, vis_model_imag, _, _ = mcmc_utils.obs_model_comparison(
            i_model, self.u, self.v, theta, d_data, self.r_out, self.nrad, self.dpix
        )
        self.assertEqual(h_mat.shape, (self.n_vis, self.nrad))
        self.assertEqual(q_eff.shape, (self.n_vis,))
        self.assertEqual(d_real_mod.shape, (self.n_vis,))
        self.assertEqual(d_imag_mod.shape, (self.n_vis,))
        self.assertEqual(vis_model.shape, (self.n_vis,))
        self.assertEqual(vis_model_imag.shape, (self.n_vis,))


if __name__ == "__main__":
    unittest.main()
