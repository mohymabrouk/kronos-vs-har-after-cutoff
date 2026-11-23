"""Command line: python -m kvh.cli <stage>.

Stages, in order: download, build, throughput, kronos, validate, freeze, test, report.
Each writes its outputs to files, so a stage can be rerun alone. `kronos` is resumable (one parquet per
model x symbol x month). `test` runs once: it refuses to run again unless --force (which is logged).
"""

import argparse
import logging
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

from kvh import config  # noqa: E402


def setup_logging(stage: str) -> None:
    config.LOGS.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(config.LOGS / f"{stage}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers, force=True)


def set_threads() -> None:
    import torch
    torch.set_num_threads(config.THREADS)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="kvh")
    p.add_argument("stage", choices=["download", "build", "throughput", "kronos", "validate", "distill", "freeze", "test", "report",
                                     "tsfm", "tsfm-validate", "tsfm-test"])
    p.add_argument("--force", action="store_true", help="test / tsfm-test stage: allow a second (logged) run")
    p.add_argument("--models", default=None, help="kronos stage: Kronos sizes (default mini,small); tsfm stage: ttm,timesfm,chronos2")
    p.add_argument("--symbols", default=None, help="kronos / tsfm stage: comma-separated symbols (default: all)")
    p.add_argument("--period", default="val", help="kronos stage: val | test | distill | slice; tsfm stage: val | test")
    p.add_argument("--tag", default=None, help="kronos stage: sampling-setting tag from config (default: frozen)")
    p.add_argument("--stride", type=int, default=1, help="kronos stage: forecast every k-th day")
    p.add_argument("--specs", default="H,D", help="tsfm stage: input specifications (PLAN.md 13.2)")
    a = p.parse_args(argv)
    setup_logging(f"{a.stage}_{a.period}" if a.stage in ("kronos", "tsfm") else a.stage)
    set_threads()
    if a.stage == "download":
        from kvh import download
        download.run()
    elif a.stage == "build":
        from kvh import data
        data.build_all()
    elif a.stage == "throughput":
        from kvh import kronos_run
        kronos_run.throughput()
    elif a.stage == "kronos":
        from kvh import kronos_run
        syms = a.symbols.split(",") if a.symbols else None
        kronos_run.run(models=(a.models or "mini,small").split(","), symbols=syms, period=a.period, tag=a.tag, stride=a.stride)
    elif a.stage == "tsfm":
        from kvh import tsfm
        syms = a.symbols.split(",") if a.symbols else None
        tsfm.run(models=(a.models or ",".join(config.TSFM_MODELS)).split(","), period=a.period, specs=a.specs.split(","), symbols=syms)
    elif a.stage == "tsfm-validate":
        from kvh import tsfm_eval
        tsfm_eval.validate()
    elif a.stage == "tsfm-test":
        from kvh import tsfm_eval
        tsfm_eval.test(force=a.force)
    elif a.stage == "validate":
        from kvh import pipeline
        pipeline.validate()
    elif a.stage == "distill":
        from kvh import student
        student.validate_and_fit()
    elif a.stage == "freeze":
        from kvh import pipeline
        pipeline.freeze()
    elif a.stage == "test":
        from kvh import pipeline
        pipeline.test(force=a.force)
    elif a.stage == "report":
        from kvh import report
        report.run()


if __name__ == "__main__":
    main()
