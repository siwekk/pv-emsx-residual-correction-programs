## The paper

**Probabilistic Residual Correction of Issued Photovoltaic Forecast Profiles without Plant Metadata**

by Krzysztof Siwek

submitted to MDPI Photovoltaics

## **EMSx reproducibility programs**

This Git repository contains source programs, dependency specifications, frozen experiment protocols, and selected aggregate results. 

The included result summaries omit local paths and commands. The public EMSx release is identified by [https://doi.org/10.5281/zenodo.5510400](https://doi.org/10.5281/zenodo.5510400).

## **Working directory and core environment**

The commands below are intended for a Linux compute host. Set the working directory to a location outside this repository. Programs with explicit input and output arguments accept any external paths. Older programs that use the common directory layout read `EMSX_WORK_ROOT` and default to a sibling directory named `pv-emsx-residual-correction-work`.

  
    `export EMSX_WORK_ROOT=/absolute/path/to/emsx-work`  
    `mkdir -p "$EMSX_WORK_ROOT"`  
    `python3 -m venv "$EMSX_WORK_ROOT/.venv-core"`  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python" -m pip install -r requirements.txt`  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python" scripts/download_emsx.py`  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python" scripts/build_emsx_daily_curves.py`   
    `--raw-root "$EMSX_WORK_ROOT/data/raw/emsx"`   
    `--output-root "$EMSX_WORK_ROOT/data/processed/emsx/daily_curves"`   
    `--manifest "$EMSX_WORK_ROOT/results/emsx_daily_curves_manifest.json"`  
 

The download program verifies every EMSx file against the checksums supplied by the Zenodo record. The curve builder creates the daily sample and its chronological partitions. It writes only to the external working directory in this example.

## **Conventional comparisons**

The repository includes the vendor forecast, persistence, seasonal naive, ARIMA, LightGBM, random forest, XGBoost, neural network, and probabilistic evaluation programs. The following commands show the daily sample baseline and residual LightGBM calculations. The remaining programs expose their inputs through `--help` or use `EMSX_WORK_ROOT`.

  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python" scripts/run_emsx_daily_baselines.py`   
    `--curve-root "$EMSX_WORK_ROOT/data/processed/emsx/daily_curves"`   
    `--prediction-output "$EMSX_WORK_ROOT/predictions/daily_baselines.parquet"`   
    `--summary-output "$EMSX_WORK_ROOT/results/daily_baselines.json"`  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python" scripts/run_emsx_daily_lightgbm.py`   
    `--curve-root "$EMSX_WORK_ROOT/data/processed/emsx/daily_curves"`   
    `--prediction-output "$EMSX_WORK_ROOT/predictions/daily_lightgbm.parquet"`   
    `--summary-output "$EMSX_WORK_ROOT/results/daily_lightgbm.json"`  
 

## **Chronos 2 and Moirai 2 repetitions**

Use separate Python environments for the two foundation models. The corresponding requirement files record the package versions used for the experiments. Install a PyTorch build suitable for the available accelerator if the default wheel does not match the host. Cache the exact model revisions named in the protocol before running the programs, since inference and adaptation load the models from the local cache. The frozen protocol specifies 32 seeds, two input variants for each model, calibration as the additional temporal evaluation, and a separate test evaluation. Do not select a favorable seed after inspecting outcomes.

  
    `export CHRONOS_PYTHON=/absolute/path/to/chronos-venv/bin/python`  
    `export MOIRAI_PYTHON=/absolute/path/to/moirai-venv/bin/python`  
    `export EMSX_RAW_ROOT="$EMSX_WORK_ROOT/data/raw/emsx"`  
    `export EMSX_CURVE_ROOT="$EMSX_WORK_ROOT/data/processed/emsx/daily_curves"`  
    `export EMSX_OUTPUT_ROOT="$EMSX_WORK_ROOT/frozen-confirmation"`  
    `bash scripts/run_emsx_frozen_confirmation_server.sh`  
 

The launcher saves per run summaries, predictions, and checkpoints only under `EMSX_OUTPUT_ROOT`. It resumes completed fits. The final summary uses every seed. The original test period informed configuration selection, so the test comparison should be interpreted descriptively. Calibration was not used to choose these configurations.

To compare the frozen forecasts at each delivery step, run the following analysis. Its tables and figure are written outside this repository.

  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python"`   
    `scripts/analyse_emsx_foundation_step_errors.py`   
    `--protocol protocols/emsx_frozen_fewshot_repeats_20260924.json`   
    `--run-root "$EMSX_OUTPUT_ROOT"`   
    `--output-root "$EMSX_WORK_ROOT/foundation-step-analysis"`  
 

## **Moirai horizon and residual diagnostic**

The separate protocol `emsx_moirai_96_residual_20260925.json` tests four arms with eight fixed seeds. It compares direct photovoltaic targets and residuals relative to the issued vendor forecast, each with and without a separate vendor covariate channel. All arms use a 96 step adaptation loss. The released recursive decoder modifies its scale in place, which prevents gradients through the predictions fed back as context. The diagnostic therefore stops gradients through those feedback values while leaving the forward predictions unchanged. It is an exploratory adaptation method, not the unmodified released training procedure. 

  
    `"$MOIRAI_PYTHON" scripts/run_emsx_moirai_96_residual.py`   
    `--protocol protocols/emsx_moirai_96_residual_20260925.json`   
    `--curve-root "$EMSX_CURVE_ROOT" --raw-root "$EMSX_RAW_ROOT"`   
    `--output-root "$EMSX_WORK_ROOT/moirai-96"`  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python"`   
    `scripts/analyse_emsx_moirai_96_residual.py`   
    `--protocol protocols/emsx_moirai_96_residual_20260925.json`   
    `--new-run-root "$EMSX_WORK_ROOT/moirai-96"`   
    `--frozen-run-root "$EMSX_OUTPUT_ROOT"`   
    `--output-root "$EMSX_WORK_ROOT/moirai-96-analysis"`  
 

## Independent checks

The following program reads externally stored prediction files, recomputes every reported metric, checks delivery keys, verifies model revisions and seeds against the frozen protocol, and confirms that the calibration and test profiles do not overlap. It exits with a nonzero status when a check fails.

  
    `"$EMSX_WORK_ROOT/.venv-core/bin/python" scripts/verify_frozen_results.py`   
    `--protocol protocols/emsx_frozen_fewshot_repeats_20260924.json`   
    `--run-root "$EMSX_OUTPUT_ROOT"`   
    `--output "$EMSX_WORK_ROOT/results/frozen-verification.json"`  
 

The programs `audit_emsx_daily_point_predictions.py` and `audit_reproducibility.py` provide additional checks for saved predictions and the earlier EMSx forecast contract. Their output paths should likewise point outside this repository. The analysis programs do not transfer data to a remote service.

## Included aggregate results

The `results/article_summaries` directory contains compact metric summaries for the point, probabilistic, robustness, and benchmark analyses. The `results/frozen` directory contains the frozen baseline summaries and the 32 seed foundation model comparison. The `results/moirai_diagnostics` directory contains stepwise errors for the original 32 seed forecasts. The `results/moirai_96` directory contains all seed level and delivery step summaries for the eight seed horizon and residual comparison, together with its figure. These are aggregate outputs for checking the claims and rebuilding the displayed comparisons. Runtime paths and command lines have been removed from the JSON summaries.

The repository excludes source measurements, full per profile forecasts, and model checkpoints. The latter prediction files and checkpoints occupied about 3.3 GB for the 96 step diagnostic alone. The scripts can regenerate these files when run with the EMSx data and the specified model checkpoints.
