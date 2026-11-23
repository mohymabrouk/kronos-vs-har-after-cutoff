#!/bin/bash
# PLAN.md section 13, after `tsfm-validate` has written results/frozen_tsfm.json (+ sha256): test-period
# forecasts with the frozen settings, each model only from its scoring start, one model at a time; then the
# single scoring run (tsfm-test, logged as an added run) and the report. Prints "done" or "FAIL <step>" to
# logs/tsfm_test_chain.out.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=logs/tsfm_test_chain.out
mkdir -p logs
step() {
  name=$1; shift
  echo "start $name $(date -u +%FT%TZ)" >> $OUT
  if ! "$@" > "logs/tsfm_test_$name.out" 2>&1; then echo "FAIL $name" >> $OUT; exit 1; fi
}
step ttm uv run python -m kvh.cli tsfm --period test --models ttm
step timesfm uv run python -m kvh.cli tsfm --period test --models timesfm
step chronos2 uv run python -m kvh.cli tsfm --period test --models chronos2
step score uv run python -m kvh.cli tsfm-test
step report uv run python -m kvh.cli report
echo "done $(date -u +%FT%TZ)" >> $OUT
