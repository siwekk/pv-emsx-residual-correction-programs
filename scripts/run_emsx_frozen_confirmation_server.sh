#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_root="${EMSX_RAW_ROOT:?Set EMSX_RAW_ROOT to the directory containing the EMSx CSV files}"
output_root="${EMSX_OUTPUT_ROOT:?Set EMSX_OUTPUT_ROOT to a directory outside this Git repository}"
chronos_python="${CHRONOS_PYTHON:?Set CHRONOS_PYTHON to the Chronos environment Python executable}"
moirai_python="${MOIRAI_PYTHON:?Set MOIRAI_PYTHON to the Moirai environment Python executable}"
curve_root="${EMSX_CURVE_ROOT:-$output_root/data/processed/emsx/daily_curves}"
protocol="$repo/protocols/emsx_frozen_fewshot_repeats_20260924.json"
output_root="$(realpath -m "$output_root")"

case "$output_root/" in
  "$repo/"*) echo "EMSX_OUTPUT_ROOT must be outside the Git repository" >&2; exit 2 ;;
esac
mkdir -p "$output_root"
exec >>"$output_root/server_run.log" 2>&1

on_exit() {
    code=$?
    printf '%s exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >"$output_root/server_status.txt"
    exit "$code"
}
trap on_exit EXIT

cd "$repo"
printf '%s frozen_confirmation_started\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$chronos_python" -u scripts/run_emsx_frozen_fewshot_repeats.py \
    --model chronos2 \
    --protocol "$protocol" \
    --curve-root "$curve_root" \
    --raw-root "$raw_root" \
    --output-root "$output_root"

"$moirai_python" -u scripts/run_emsx_frozen_fewshot_repeats.py \
    --model moirai2 \
    --protocol "$protocol" \
    --curve-root "$curve_root" \
    --raw-root "$raw_root" \
    --output-root "$output_root"

"$chronos_python" -u scripts/summarize_emsx_frozen_repeats.py \
    --protocol "$protocol" \
    --output-root "$output_root"

printf '%s frozen_confirmation_finished\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
