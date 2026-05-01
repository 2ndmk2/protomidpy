import argparse
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_calc import build_model_products


def _infer_spw_visfile(visfile):
    visfile_path = Path(visfile).resolve()
    target_id = visfile_path.name.replace("_continuum_averaged.vis.npz", "")
    candidate = TESTS_DIR / "vis_data" / f"{target_id}_vis_each_spw.npz"
    if candidate.exists():
        return str(candidate)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample_for_rad", type=int, default=20)
    parser.add_argument("--n_burnin", type=int, default=20000)
    parser.add_argument("--mcmc_result_file", default=str(TESTS_DIR / "result_warp" / "AS209_continuum_averaged.vis_mcmc.npz"))
    parser.add_argument("--visfile", default=str(TESTS_DIR / "AS209_continuum_averaged.vis.npz"))
    parser.add_argument("--out_file_for_model", default=str(TESTS_DIR / "result_warp" / "AS209_continuum_averaged_model.npz"))
    parser.add_argument(
        "--spw_visfile",
        default="auto",
        help="npz created by ms_to_npz_for_spw.py. Use 'auto' to load ./vis_data/<target>_vis_each_spw.npz when it exists.",
    )
    args = parser.parse_args()
    spw_visfile = args.spw_visfile
    if spw_visfile == "auto":
        spw_visfile = _infer_spw_visfile(args.visfile)

    build_model_products(
        args.mcmc_result_file,
        args.visfile,
        args.out_file_for_model,
        n_sample_for_rad=args.n_sample_for_rad,
        n_burnin=args.n_burnin,
        spw_visfile=spw_visfile,
    )


if __name__ == "__main__":
    main()
