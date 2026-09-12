# -*- coding: utf-8 -*-
"""无绘图复现 Q1--Q4 的 L1 Lite 主配置结果。"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = Path(__file__).resolve().parent


def commands() -> list[list[str]]:
    py = sys.executable
    return [
        [py, str(SCRIPT_DIR / "问题1_求解.py"), "--no-figures"],
        [py, str(SCRIPT_DIR / "问题2_求解.py"), "--main-only", "--days", "365", "--executor", "mpc"],
        [py, str(SCRIPT_DIR / "问题3_求解.py"), "--days", "365", "--workers", "1", "--main-only"],
        [py, str(SCRIPT_DIR / "问题4_求解.py"), "--days", "365", "--workers", "2", "--main-only"],
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列出复现步骤，不执行")
    args = ap.parse_args()
    jobs = commands()
    if args.list:
        for job in jobs:
            print(subprocess.list2cmdline(job))
        return 0
    started = time.perf_counter()
    for i, job in enumerate(jobs, 1):
        print(f"[Lite {i}/{len(jobs)}] {subprocess.list2cmdline(job)}", flush=True)
        subprocess.run(job, cwd=ROOT, check=True)
    print(f"[Lite] all done, elapsed_s={time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
