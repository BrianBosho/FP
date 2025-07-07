#!/bin/bash

BASE_BRANCH="pubmed_fix"
BRANCHES=(
  "main"
  "minimal"
  "minimal_backup"
  "lean"
  "feat/restructure"
  "feat-lean/pe"
  "fixes/main_minimal"
  "integrate-minimal"
  "ref/phase1"
)

OUTPUT_FILE="branch_diff_report.txt"
echo "Branch Comparison Report (vs '$BASE_BRANCH')" > $OUTPUT_FILE
echo "Generated on $(date)" >> $OUTPUT_FILE
echo "==================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for BR in "${BRANCHES[@]}"; do
  echo "🔍 Branch: $BR" >> $OUTPUT_FILE
  echo "----------------------------------------" >> $OUTPUT_FILE

  if git show-ref --verify --quiet refs/heads/$BR; then
    COMMITS=$(git log $BASE_BRANCH..$BR --oneline 2>/dev/null)

    if [ -z "$COMMITS" ]; then
      echo "✅ No unique commits in '$BR' (fully merged into $BASE_BRANCH or identical)" >> $OUTPUT_FILE
    else
      echo "❗ Unique commits in '$BR' not in '$BASE_BRANCH':" >> $OUTPUT_FILE
      echo "$COMMITS" | sed 's/^/   - /' >> $OUTPUT_FILE
    fi
  else
    echo "⚠️ Branch '$BR' not found in local repo." >> $OUTPUT_FILE
  fi

  echo "" >> $OUTPUT_FILE
done

echo "✅ Done. Report saved to: $OUTPUT_FILE"
