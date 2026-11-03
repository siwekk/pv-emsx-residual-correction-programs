"""Portable input and output locations for the November 2026 experiments."""
import os
from pathlib import Path
from emsx_paths import CODE_ROOT, WORK_ROOT

ROOT=Path(os.environ.get('EMSX_ROUND1_ROOT',WORK_ROOT/'round1')).resolve()
CURVE_ROOT=Path(os.environ.get('EMSX_CURVE_ROOT',WORK_ROOT/'data/processed/emsx/daily_curves')).resolve()
PREDICTION_ROOT=Path(os.environ.get('EMSX_PREDICTION_ROOT',WORK_ROOT/'predictions')).resolve()
if ROOT==CODE_ROOT or CODE_ROOT in ROOT.parents:
    raise RuntimeError('EMSX_ROUND1_ROOT must be outside the program repository')
