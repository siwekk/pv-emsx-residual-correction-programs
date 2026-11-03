#!/usr/bin/env bash
# Starts round2_compute.py in the background on the server and writes a log.
# Usage from ~/emsx_round2:  bash scripts/run_round2.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p results
nohup python3 scripts/round2_compute.py \
  --curves data/daily_curves \
  --original-predictions data/emsx_daily_lightgbm.parquet \
  --out results > results/round2_log.txt 2>&1 &
echo "Started, PID $!. Progress: tail -f ~/emsx_round2/results/round2_log.txt"
