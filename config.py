"""
config.py
---------
Central configuration. Nothing else in the repo hardcodes paths, seeds, or grids.
"""
from pathlib import Path

# paths
ROOT        = Path.home() / "zero-inflated-benchmark"
DATA_DIR    = ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
CSV_DIR     = RESULTS_DIR / "csv"
LOG_DIR     = RESULTS_DIR / "logs"
FIGURES_DIR = ROOT / "figures"

for _d in (RAW_DIR, PROC_DIR, CSV_DIR, LOG_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# seeds
MASTER_SEED = 20261015

# simulation grids
STRUCT_ZERO_GRID   = [0.0, 0.3, 0.6]
DEPENDENCE_GRID    = ["none", "occurrence", "magnitude", "both"]
NONLINEARITY_GRID  = ["linear", "quadratic", "threshold"]
SAMPLE_SIZE_GRID   = [300, 600]

# splits
INITIAL_TRAIN_FRAC = 0.6
VAL_FRAC           = 0.1
HORIZON            = 1

# diagnostics
ZERO_DIAG_MIN_POS  = 10
DEP_DIAG_HOLDOUT   = 0.2

# neural
NEURAL_EPOCHS      = 200
NEURAL_PATIENCE    = 30
NEURAL_BATCH       = 32
NEURAL_LR          = 1e-3
NEURAL_HIDDEN      = 64
# Pilot
PILOT_N_REPS       = 20