import argparse
import os
import sys
from pathlib import Path
ARCSEC_TO_RAD= 1/206265.0
os.environ["OMP_NUM_THREADS"] = "3"
from multiprocessing import Pool, freeze_support

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protomidpy import main
from protomidpy import mcmc_utils
from protomidpy import utils

parser = argparse.ArgumentParser() 
parser.add_argument('--n_process', default=3)
parser.add_argument('--config', default=str(TESTS_DIR / 'paras' / 'mcmc_config.dat'))
parser.add_argument('--initial_para', default=str(TESTS_DIR / 'paras' / 'AS209_paradic.dat'))
parser.add_argument('--prior', default=str(TESTS_DIR / 'paras' / 'prior.dat'))
parser.add_argument('--visfile', default=str(TESTS_DIR / 'AS209_continuum_averaged.vis.npz'))


if __name__ == '__main__':

    args = parser.parse_args()
    mcmc_config= args.config
    mcmc_initial_para= args.initial_para
    mcmc_prior= args.prior
    visfile = args.visfile
    n_process = int(args.n_process)

    para_dic_for_mcmc_config =mcmc_utils.make_para_dic_for_mcmc(mcmc_config)
    para_dic_for_mcmc_initial =mcmc_utils.make_para_dic_for_mcmc(mcmc_initial_para)
    para_dic_for_mcmc_prior =mcmc_utils.make_para_dic_for_mcmc(mcmc_prior)

    header_name_for_file = visfile.split("/")[-1].replace(".npz", "")
    u_d, v_d, vis_d, wgt_d, freq_d = utils.load_obsdata(visfile)

    ## Multi-proceessing for MCMC
    with Pool(processes=n_process) as pool:
        main.sample_mcmc_full(u_d, v_d, vis_d, wgt_d, "RBF", -1000,  para_dic_for_mcmc_config["Nwalker"], para_dic_for_mcmc_config["Nchain"], para_dic_for_mcmc_prior, para_dic_for_mcmc_initial, \
            header_name_for_file = header_name_for_file, out_dir = para_dic_for_mcmc_config["Out_folder"], n_bin_log = para_dic_for_mcmc_config["Nbin"], \
            nrad=para_dic_for_mcmc_config["Nrad"], dpix= para_dic_for_mcmc_config["Dpix"]* ARCSEC_TO_RAD, q_min_max_bin = [ para_dic_for_mcmc_config["Qmin"],  para_dic_for_mcmc_config["Qmax"]] , pool =pool)


