#!/bin/bash
# PLAN.md section 13: validation-period forecasts of the three newer foundation models (every day, 10 assets,
# both input specifications, both hourly variants), one model at a time. `tsfm-validate` (light) follows
# separately. Prints "done" or "FAIL <step>" to logs/tsfm_val_chain.out.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=logs/tsfm_val_chain.out
mkdir -p logs
step() {
  name=$1; shift
  echo "start $name $(date -u +%FT%TZ)" >> $OUT
  if ! "$@" > "logs/tsfm_val_$name.out" 2>&1; then echo "FAIL $name" >> $OUT; exit 1; fi
}
step ttm uv run python -m kvh.cli tsfm --period val --models ttm
step timesfm uv run python -m kvh.cli tsfm --period val --models timesfm
step chronos2 uv run python -m kvh.cli tsfm --period val --models chronos2
echo "done $(date -u +%FT%TZ)" >> $OUT
