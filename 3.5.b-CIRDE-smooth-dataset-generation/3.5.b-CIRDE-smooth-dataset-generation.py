#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the three dataset scripts sequentially:

    1. make_data(1).py
    2. merge_npz.py
    3. check_npz.py

The pipeline stops immediately if any stage fails.

Run with

    python run_dataset_pipeline.py

The three scripts must be in the same directory as this file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ============================================================
# Pipeline configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = (
    "make_data(1).py",
    "merge_npz.py",
    "check_npz.py",
)


# ============================================================
# Helpers
# ============================================================

def run_script(filename: str) -> None:
    """
    Run one Python script using the same Python interpreter that
    launched this pipeline.
    """

    path = SCRIPT_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Required script not found: {path}"
        )

    separator = "=" * 72

    print()
    print(separator)
    print(f"RUNNING: {filename}")
    print(separator)
    print()

    subprocess.run(
        [sys.executable, str(path)],
        cwd=SCRIPT_DIR,
        check=True,
    )

    print()
    print(f"FINISHED: {filename}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Dataset pipeline")
    print("----------------")
    print(f"Python      : {sys.executable}")
    print(f"Working dir : {SCRIPT_DIR}")

    try:
        for filename in SCRIPTS:
            run_script(filename)

    except subprocess.CalledProcessError as exc:
        print()
        print("PIPELINE FAILED")
        print(f"Script exited with return code {exc.returncode}.")
        sys.exit(exc.returncode)

    except FileNotFoundError as exc:
        print()
        print("PIPELINE FAILED")
        print(exc)
        sys.exit(1)

    print()
    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()