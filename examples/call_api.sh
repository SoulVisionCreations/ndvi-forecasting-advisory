#!/usr/bin/env bash
# Smoke-test a running serve_api (default http://localhost:8001).
# Start the server first (see the repo README / examples/README.md), then run this.
#   bash examples/call_api.sh                       # localhost:8001
#   bash examples/call_api.sh http://HOST:PORT      # somewhere else
set -euo pipefail

HOST="${1:-http://localhost:8001}"
REQ="$(dirname "$0")/forecast_request.json"

echo "== GET ${HOST}/health =="
curl -s "${HOST}/health"; echo

echo
echo "== POST ${HOST}/forecast  (body = forecast_request.json) =="
curl -s -X POST "${HOST}/forecast" \
  -H "Content-Type: application/json" \
  --data @"${REQ}"; echo

echo
echo "== POST ${HOST}/forecast?debug=true  (verbose) =="
curl -s -X POST "${HOST}/forecast?debug=true" \
  -H "Content-Type: application/json" \
  --data @"${REQ}"; echo
