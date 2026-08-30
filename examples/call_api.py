#!/usr/bin/env python3
"""Call a running serve_api over the sample points and print whatever it returns.

Usage:
    python examples/call_api.py                    # http://localhost:8001
    python examples/call_api.py http://HOST:PORT

There are no "expected" values here -- this just confirms the server answers with a
populated forecast for each sample point. Numbers depend on live satellite/weather data.
"""
import csv
import os
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
HERE = os.path.dirname(os.path.abspath(__file__))

print("health:", requests.get(f"{BASE}/health", timeout=10).json())

with open(os.path.join(HERE, "sample_points.csv")) as f:
    for row in csv.DictReader(f):
        body = {"lat": float(row["lat"]), "lon": float(row["lon"]), "date": row["date"]}
        r = requests.post(f"{BASE}/forecast", json=body, timeout=180)   # first call is slow (~30-60s)
        tag = f"{row['lat']},{row['lon']} {row['date']}"
        if r.status_code == 200:
            d = r.json()
            print(f"{tag} -> NDVI {d['forecast_ndvi']} | "
                  f"{d['relative_vegetation_condition']} | target {d['target_date']}")
        else:
            print(f"{tag} -> HTTP {r.status_code}: {r.text}")
