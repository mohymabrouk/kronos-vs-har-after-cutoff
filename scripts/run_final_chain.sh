#!/bin/bash
# Final stages, after the Kronos chain: student (distill), freeze, one-shot test scoring, report.
set -e
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /c/dev/quant-research/project_3_kronos_vs_har
echo "start $(date)" >> logs/final_chain.out
uv run python -m kvh.cli distill > logs/distill.out 2>&1
echo "distill done $(date)" >> logs/final_chain.out
uv run python -m kvh.cli freeze > logs/freeze.out 2>&1
echo "freeze done $(date)" >> logs/final_chain.out
uv run python -m kvh.cli test > logs/test.out 2>&1
echo "test done $(date)" >> logs/final_chain.out
uv run python -m kvh.cli report > logs/report.out 2>&1
echo "report done $(date)" >> logs/final_chain.out
echo FINAL_DONE >> logs/final_chain.out
