#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=/home/bosho/.conda/envs/fedgnn/bin/python
BASE_CFG="$REPO_ROOT/experiments/configs/scalability/S1_full_concurrency_sweep_base.yaml"
GEN_CFG_DIR="$REPO_ROOT/experiments/configs/scalability/generated"
LOG_DIR="$REPO_ROOT/logs/scalability/full_concurrency_sweep"
mkdir -p "$GEN_CFG_DIR" "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/sweep_resume_5_10_${TS}_summary.tsv"
echo -e "run_ts\ttarget_concurrency\tstatus\tlog\tconfig\tresource_csv" > "$SUMMARY"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/home/bosho/.conda/envs/fedgnn/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

make_cfg() {
  local n="$1"
  local cfg="$GEN_CFG_DIR/S1_full_concurrency_N${n}_${TS}.yaml"
  "$PYTHON" - <<PY
from pathlib import Path
import yaml
base = yaml.safe_load(Path(r"$BASE_CFG").read_text())
base['max_concurrent_clients'] = int($n)
base['results_dir'] = f"experiments/results/scalability/full_concurrency_sweep/N{int($n)}"
Path(r"$cfg").write_text(yaml.safe_dump(base, sort_keys=False))
print(r"$cfg")
PY
}

monitor_resources() {
  local csv="$1"
  while true; do
    ts_now="$(date +%Y-%m-%dT%H:%M:%S)"
    mem_line="$(free -b | awk '/Mem:/ {print $2","$3","$4","$7}')"
    gpu_line="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    proc_line="$(ps -eo pid,rss,%mem,%cpu,comm --sort=-rss | head -6 | tail -5 | tr '\n' ';' | sed 's/;$/\n/')"
    echo "$ts_now,$mem_line,$gpu_line,\"$proc_line\"" >> "$csv"
    sleep 20
  done
}

run_one() {
  local n="$1"
  local cfg
  cfg="$(make_cfg "$n" | tail -1)"
  local log="$LOG_DIR/N${n}_${TS}.log"
  local csv="$LOG_DIR/N${n}_${TS}_resources.csv"
  echo 'timestamp,mem_total_b,mem_used_b,mem_free_b,mem_available_b,gpu_mem_used_mb,gpu_mem_total_mb,gpu_util_pct,top_procs' > "$csv"

  echo "=== Starting concurrency test N=$n ===" | tee -a "$log"
  echo "Config: $cfg" | tee -a "$log"
  ray stop --force >/dev/null 2>&1 || true
  monitor_resources "$csv" &
  local mon_pid=$!

  set +e
  timeout 45m "$PYTHON" -m src.fedgnn.experiments.run_experiments --config "$cfg" >> "$log" 2>&1
  local rc=$?
  set -e

  kill "$mon_pid" >/dev/null 2>&1 || true
  wait "$mon_pid" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true

  local status="ok"
  if [[ $rc -eq 124 ]]; then
    status="timeout"
  elif [[ $rc -ne 0 ]]; then
    status="exit_$rc"
  fi
  if grep -Eqi 'OutOfMemory|CUDA out of memory|memory pressure|Workers .* killed due to memory pressure|cannot be scheduled right now|RuntimeError' "$log"; then
    if grep -Eqi 'OutOfMemory|CUDA out of memory|memory pressure|Workers .* killed due to memory pressure' "$log"; then
      status="oom_or_pressure"
    elif grep -Eqi 'cannot be scheduled right now' "$log"; then
      status="sched_pressure"
    fi
  fi

  echo -e "$TS\t$n\t$status\t$log\t$cfg\t$csv" >> "$SUMMARY"
  echo "=== Finished N=$n status=$status ===" | tee -a "$log"
}

for n in 5 6 7 8 9 10; do
  run_one "$n"
done

echo "Resume sweep completed through N=10. Summary: $SUMMARY"
