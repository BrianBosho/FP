#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=1

usage() {
  cat <<'EOF'
Usage:
  ./scripts/clean_artifacts.sh [--apply]

Deletes only known generated artifact directories/files:
  - wandb/ runs/ results/ results_summary/ logs/ training_logs/ archive/
  - nohup*.out *.out *.log results.csv

Default mode is DRY RUN (prints what would be deleted).
Use --apply to actually delete.
EOF
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1-}" == "--apply" ]]; then
  DRY_RUN=0
elif [[ "${1-}" != "" ]]; then
  echo "Unknown argument: ${1}"
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

targets=(
  "wandb"
  "runs"
  "results"
  "results_summary"
  "logs"
  "training_logs"
  "archive"
  "nohup.out"
  "results.csv"
)

echo "Repo: ${REPO_ROOT}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Mode: DRY RUN (pass --apply to delete)"
else
  echo "Mode: APPLY (deleting)"
fi
echo

rm_path() {
  local p="$1"
  if [[ ! -e "${p}" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "would delete: ${p}"
  else
    echo "deleting: ${p}"
    rm -rf -- "${p}"
  fi
}

for t in "${targets[@]}"; do
  rm_path "${t}"
done

# Globs (only in repo root)
shopt -s nullglob
for f in nohup*.out *.out *.log; do
  rm_path "${f}"
done
shopt -u nullglob

echo
echo "Done."

