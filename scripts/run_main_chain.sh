#!/bin/bash
# Runs after `validate` has written frozen_kronos.json (mini, small). Kronos jobs only, one at a time:
#   test-period forecasts (frozen settings; mini, then small), the distillation teacher (small, pre-validation),
#   the Kronos-base validation subset, `validate` again (adds base; mini/small choices are deterministic and
#   unchanged), then Kronos-base on the test subset. The student (light) and `freeze` are run separately,
#   before `test` scores anything.
set -e
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /c/dev/quant-research/project_3_kronos_vs_har
SMALL_TAG=$(uv run python -c "import json; print(json.load(open('results/frozen_kronos.json'))['small']['tag'])")
echo "start $(date); small tag $SMALL_TAG" >> logs/chain.out
uv run python -m kvh.cli kronos --period test --models mini > logs/test_mini.out 2>&1
echo "test mini done $(date)" >> logs/chain.out
uv run python -m kvh.cli kronos --period test --models small > logs/test_small.out 2>&1
echo "test small done $(date)" >> logs/chain.out
uv run python -m kvh.cli kronos --period distill --models small --stride 6 > logs/distill_teacher.out 2>&1
echo "teacher done $(date)" >> logs/chain.out
uv run python -m kvh.cli kronos --period val --models base --symbols BTCUSDT,ETHUSDT --tag $SMALL_TAG --stride 6 > logs/val_base.out 2>&1
echo "base val done $(date)" >> logs/chain.out
uv run python -m kvh.cli validate > logs/validate2.out 2>&1
echo "validate (with base) done $(date)" >> logs/chain.out
uv run python -m kvh.cli kronos --period test --models base --symbols BTCUSDT,ETHUSDT --stride 4 > logs/test_base.out 2>&1
echo "test base done $(date)" >> logs/chain.out
echo CHAIN_DONE >> logs/chain.out
