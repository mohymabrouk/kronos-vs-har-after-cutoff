#!/bin/bash
# Waits for the validation grid, runs `validate` (frozen_kronos.json), then the main Kronos chain.
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$(dirname "$0")/.."
until grep -q VALGRID_DONE logs/val_small.out 2>/dev/null; do sleep 10; done
uv run python -m kvh.cli validate > logs/validate1.out 2>&1 || { echo "VALIDATE_FAILED" >> logs/chain.out; exit 1; }
echo "validate done $(date)" >> logs/chain.out
scripts/run_main_chain.sh || echo "CHAIN_FAILED $(date)" >> logs/chain.out
