#!/usr/bin/env python3
"""Create wsl_run_lenet5_accel_6x3_spatial.sh from the 6x3 single-pipeline script."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "testcases/lenet5/wsl_run_lenet5_accel_6x3.sh"
dst = root / "testcases/lenet5/wsl_run_lenet5_accel_6x3_spatial.sh"

content = src.read_text(encoding="utf-8")
content = content.replace(
    "# Distributed LeNet-5 on 6x3 grid: linear 6-stage pipeline on row 0 (ACCEL_MODE=hw).",
    "# Spatial batching: 2 parallel LeNet pipelines on 6x3 (rows 0 and 1).",
)
content = content.replace("lenet5_accel_6x3.yaml", "lenet5_accel_6x3_spatial.yaml")
content = content.replace("LENET5_LAYOUT=6x3", "LENET5_LAYOUT=6x3_spatial")
content = content.replace(
    "    --exclude 'testcases/lenet5/lenet5_accel_6x3' \\\n",
    "    --exclude 'testcases/lenet5/lenet5_accel_6x3' \\\n"
    "    --exclude 'testcases/lenet5/lenet5_accel_6x3_spatial' \\\n",
)
content = content.replace(
    'BATCH_SIZE="${LENET5_BATCH_SIZE:-2}"',
    'SPATIAL_P="${LENET5_SPATIAL_PIPELINES:-2}"',
)
content = content.replace(
    'GEN_FLAGS=(--batch "$BATCH_SIZE")',
    'GEN_FLAGS=(--spatial-pipelines "$SPATIAL_P")',
)
content = content.replace(
    'python3 "$HEMPS_PATH/tools/gen_lenet5_input_batch.py" "${GEN_FLAGS[@]}"',
    'python3 "$HEMPS_PATH/tools/gen_lenet5_input_batch.py" "${GEN_FLAGS[@]}"',
)
content = content.replace(
    'python3 "$HEMPS_PATH/tools/gen_lenet5_input_batch.py" --spatial-pipelines "$SPATIAL_P"',
    'python3 "$HEMPS_PATH/tools/gen_lenet5_input_batch.py" "${GEN_FLAGS[@]}"',
)
content = content.replace(
    'echo "Grid: 6x3 linear row 0 | ACCEL_MODE=$ACCEL_MODE | BATCH=$BATCH_SIZE"',
    'echo "Grid: 6x3 spatial ($SPATIAL_P pipelines) | ACCEL_MODE=$ACCEL_MODE"',
)
content = content.replace(
    "grep -E 'Simulation time|END OF ALL|LENET |\\[ACCEL\\]|IC-ALERT|expected class|predicted class|batch correct|pipeline|inject'",
    "grep -E 'Simulation time|END OF ALL|LENET |\\[ACCEL\\]|IC-ALERT|expected class|predicted class|spatial|inject|Fatal|clock_tx_io'",
)
content = content.replace(
    "if [ \"$rc\" -ne 0 ] || grep -qE 'ERROR:|Insuficient page size|\\*\\*\\* Error: hemps-run stopped'",
    "if [ \"$rc\" -ne 0 ] || grep -qE 'ERROR:|Insuficient page size|\\*\\*\\* Error: hemps-run stopped|Fatal SystemC|clock_tx_io'",
)
content = content.replace('echo "=== master PE [0,0] ==="', 'echo "=== master PE [0,0] pipeline 0 ==="')
old_tail = """echo "=== c1p1 PE [1,0] ==="
grep -E 'LENET |\\[ACCEL\\]' "$TC/$TC_NAME/log/log1x0.txt" 2>/dev/null | tail -10 || true
echo "=== fc3 PE [5,0] ==="
grep -E 'LENET |\\[ACCEL\\]' "$TC/$TC_NAME/log/log5x0.txt" 2>/dev/null | tail -10 || true

python3 "$HEMPS_PATH/tools/parse_lenet_pipeline_timing.py" "$TC/$TC_NAME/log" || true"""
new_tail = """echo "=== master PE [0,1] pipeline 1 ==="
grep -E 'LENET |predicted|expected|spatial|\\[ACCEL\\]' "$TC/$TC_NAME/log/log0x1.txt" 2>/dev/null | tail -15 || true

LENET5_LAYOUT=6x3_spatial python3 "$HEMPS_PATH/tools/parse_lenet_pipeline_timing.py" "$TC/$TC_NAME/log" || true"""
content = content.replace(old_tail, new_tail)

dst.write_text(content, encoding="utf-8", newline="\n")
print(f"Wrote {dst} ({len(content)} bytes)")
