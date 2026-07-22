#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAKE_VIS_DIR="$ROOT_DIR/working/make_vis_data"
RESID_DIR="$ROOT_DIR/working/compute_simulation_residual"
ANALYSIS_DIR="$ROOT_DIR/working/warp_analysis"

CASA_BIN="${CASA_BIN:-/Applications/CASA.app/Contents/MacOS/casa}"
MODEL_SEED="${MODEL_SEED:-1}"
RUN_NONDEPROJ_ANALYSIS="${RUN_NONDEPROJ_ANALYSIS:-0}"
ARCHIVE_PREVIOUS="${ARCHIVE_PREVIOUS:-0}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

log() {
  printf '[%s] %s\n' "$(date +"%H:%M:%S")" "$*"
}

run_casa() {
  local workdir="$1"
  local script_name="$2"
  log "CASA: ${script_name}"
  (
    cd "$workdir"
    env LC_ALL=C LANG=C "$CASA_BIN" --nologger --nogui -c "$script_name"
  )
}

archive_previous_outputs() {
  local tag
  tag="$(timestamp)"
  log "Archiving current outputs to ${tag}"
  mkdir -p \
    "$MAKE_VIS_DIR/archive_${tag}" \
    "$RESID_DIR/archive_${tag}" \
    "$ANALYSIS_DIR/archive_${tag}"

  cp -R \
    "$MAKE_VIS_DIR"/simulated_warp_model.npz \
    "$MAKE_VIS_DIR"/simulated_warp_vis.npz \
    "$MAKE_VIS_DIR"/ms_sim_data \
    "$MAKE_VIS_DIR/archive_${tag}/" 2>/dev/null || true

  cp -R \
    "$RESID_DIR"/result_fixed \
    "$RESID_DIR"/images_input \
    "$RESID_DIR"/images_input_deprojected \
    "$RESID_DIR"/images_res \
    "$RESID_DIR"/images_res_deprojected \
    "$RESID_DIR"/ms_data_sub \
    "$RESID_DIR"/ms_data_sub_deprojected \
    "$RESID_DIR/archive_${tag}/" 2>/dev/null || true

  cp -R \
    "$ANALYSIS_DIR"/result \
    "$ANALYSIS_DIR/archive_${tag}/" 2>/dev/null || true
}

cleanup_outputs() {
  log "Cleaning generated folders"
  rm -rf \
    "$MAKE_VIS_DIR/ms_sim_data" \
    "$RESID_DIR/ms_data_sub" \
    "$RESID_DIR/ms_data_sub_deprojected" \
    "$RESID_DIR/ms_input_deprojected" \
    "$RESID_DIR/images_input" \
    "$RESID_DIR/images_input_deprojected" \
    "$RESID_DIR/images_res" \
    "$RESID_DIR/images_res_deprojected"

  rm -f \
    "$RESID_DIR"/TempLattice* \
    "$RESID_DIR"/casa-*.log \
    "$MAKE_VIS_DIR"/casa-*.log
}

run_python() {
  log "Python: $*"
  (cd "$ROOT_DIR" && python "$@")
}

run_analysis() {
  log "Running deprojected warp analysis"
  run_python working/warp_analysis/compute_m2_delta_cosi.py \
    --r-min-arcsec 0.8 \
    --r-max-arcsec 1.5 \
    --out-prefix working/warp_analysis/result/simulated_warp_m2_deproj_default_r0p8_1p5

  run_python working/warp_analysis/compute_m2_delta_cosi.py \
    --i0-source model \
    --r-min-arcsec 0.8 \
    --r-max-arcsec 1.5 \
    --out-prefix working/warp_analysis/result/simulated_warp_m2_deproj_default_r0p8_1p5_modelI

  run_python working/warp_analysis/compare_i0_sources.py \
    --image-npz working/warp_analysis/result/simulated_warp_m2_deproj_default_r0p8_1p5.npz \
    --model-npz working/warp_analysis/result/simulated_warp_m2_deproj_default_r0p8_1p5_modelI.npz \
    --out-prefix working/warp_analysis/result/simulated_warp_m2_deproj_default_r0p8_1p5_compare

  if [[ "$RUN_NONDEPROJ_ANALYSIS" == "1" ]]; then
    log "Running non-deprojected warp analysis"
    run_python working/warp_analysis/compute_m2_delta_cosi.py \　
      --images-already-deprojected=false \
      --input-fits working/compute_simulation_residual/images_input/image_simulated_warp.fits \
      --residual-fits working/compute_simulation_residual/images_res/image_simulated_warp_fixed_model.fits \
      --r-min-arcsec 0.8 \
      --r-max-arcsec 1.5 \
      --out-prefix working/warp_analysis/result/simulated_warp_m2_nodeproj_r0p8_1p5
  fi
}

main() {
  [[ -x "$CASA_BIN" ]] || {
    echo "CASA binary not found: $CASA_BIN" >&2
    exit 1
  }

  if [[ "$ARCHIVE_PREVIOUS" == "1" ]]; then
    archive_previous_outputs
  fi

  cleanup_outputs

  run_python working/make_vis_data/make_warp_simulation.py
  run_casa "$MAKE_VIS_DIR" make_ms.py

  run_python working/compute_simulation_residual/model_calc_fixed.py --seed "$MODEL_SEED"
  run_casa "$RESID_DIR" sub_model.py

  log "Preparing copied deprojected MS folders"
  cp -R "$RESID_DIR/ms_data_sub" "$RESID_DIR/ms_data_sub_deprojected"
  cp -R "$MAKE_VIS_DIR/ms_sim_data/simulated_warp.ms" "$RESID_DIR/ms_input_deprojected"

  run_casa "$RESID_DIR" deprojected.py
  run_casa "$RESID_DIR" image_input_ms.py
  run_casa "$RESID_DIR" image_input_ms_deprojected.py
  run_casa "$RESID_DIR" all_tclean_res.py
  run_casa "$RESID_DIR" all_tclean_res_deprojected.py

  run_analysis

  log "Pipeline finished"
}

main "$@"
