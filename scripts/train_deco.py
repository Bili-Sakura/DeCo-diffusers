#!/usr/bin/env python3
"""Train a class-conditioned DeCo model."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parent
for path in (REPO_SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deco_training.cli import main

if __name__ == "__main__":
    main()
