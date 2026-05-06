#!/bin/sh
MON=/home/bosho/FP/experiments/propagator_eval/MONITORING.md
LOG=/home/bosho/FP/experiments/propagator_eval/results/phase_1_cora_intrinsic/logs/phase1__cora__diffusion__beta10000__seed0.log
PID=524446
COUNT=0
while [ $COUNT -lt 12 ]; do
  TS=$(date '+%Y-%m-%d %H:%M %Z')
  if [ -d /proc/$PID ]; then PSTATUS=running; else PSTATUS=finished; fi
  GPU=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | awk -F', ' '{printf "%s MiB used / %s MiB free / %s%% util", $1,$2,$3}')
  RAM=$(free -h | awk '/Mem:/ {printf "%s used / %s free / %s avail", $3,$4,$7}')
  LOAD=$(uptime | sed 's/.*load average: //')
  LAST=$(tail -n 3 "$LOG" 2>/dev/null | sed 's/|/\\|/g')
  {
    printf '\n### %s\n\n' "$TS"
    printf '| Item | Status | Notes |\n'
    printf '|---|---|---|\n'
    printf '| Smoke run process | %s | pid %s |\n' "$PSTATUS" "$PID"
    printf '| GPU | snapshot | %s |\n' "$GPU"
    printf '| RAM | snapshot | %s |\n' "$RAM"
    printf '| Load | snapshot | %s |\n' "$LOAD"
    printf '| Log tail | snapshot | %s |\n' "$(printf '%s' "$LAST" | tr '\n' ' ' )"
  } >> "$MON"
  [ "$PSTATUS" = finished ] && exit 0
  COUNT=$((COUNT+1))
  sleep 600
 done
