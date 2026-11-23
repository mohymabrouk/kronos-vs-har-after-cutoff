#!/bin/bash
# Validation grid for Kronos sampling settings (see PLAN.md section 3 for the arithmetic).
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$(dirname "$0")/.."
uv run python -m kvh.cli kronos --period val --models mini --tag T06,T09,T10 --stride 6 > logs/val_mini.out 2>&1
uv run python -m kvh.cli kronos --period val --models small --tag T06,T09,T10 --stride 9 > logs/val_small.out 2>&1
echo VALGRID_DONE >> logs/val_small.out
