"""Resolve data and output locations outside the source repository."""

from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get(
    "EMSX_WORK_ROOT", CODE_ROOT.parent / "pv-emsx-residual-correction-work"
)).resolve()

if WORK_ROOT == CODE_ROOT or CODE_ROOT in WORK_ROOT.parents:
    raise RuntimeError("EMSX_WORK_ROOT must be outside the source repository")
