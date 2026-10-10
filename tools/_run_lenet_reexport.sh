#!/bin/bash
set -euo pipefail
cd "/mnt/c/Users/mohamed elreedy/Downloads/hemps_OSZ-MScVersion_Modified"

if python3 -c "import numpy" 2>/dev/null; then
  PY=python3
else
  python3 -m venv /tmp/lenet_venv
  /tmp/lenet_venv/bin/pip install -q numpy
  PY=/tmp/lenet_venv/bin/python
fi
echo "Using $PY"
$PY -c "import numpy; print('numpy', numpy.__version__)"

export PYTHONPATH=tools
echo "=== export lenet5 assets ==="
$PY tools/export_lenet5_assets.py --epochs 30 --train-limit 10000 --lr 0.05 --min-test-acc 0.55

echo "=== gen 100-batch header ==="
$PY tools/gen_lenet5_tb_batch.py --batch 100 --split test

echo "=== SW pred check ==="
$PY tools/_check_lenet_sw_batch.py

echo "=== SystemC TB ==="
cd hardware/sc/tb/tensor_accel_unit
make clean
make run
