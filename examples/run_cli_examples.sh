#!/usr/bin/env bash
# Run the inference CLI over the sample points.
# Run from anywhere; assumes the model asset is unzipped into <repo>/weights/.
#   bash examples/run_cli_examples.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"

tail -n +2 examples/sample_points.csv | while IFS=, read -r lat lon date; do
  [ -z "${lat}" ] && continue
  echo "======== ${lat},${lon}  ${date} ========"
  python inference/infer_cli.py --lat "${lat}" --lon "${lon}" --date "${date}" \
    --run_dir weights --model_dir weights \
    --lookup_csv weights/mws_static_lookup_UNSCALED.tsv
done
