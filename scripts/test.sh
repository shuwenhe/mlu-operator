#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/build.sh cpu
artifacts/build/gemm_reference | tee artifacts/accuracy/gemm_cpu_reference.csv
