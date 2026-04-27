#!/usr/bin/env python3
"""Add device configuration to all YAML configs."""

from pathlib import Path
import re

CONFIGS_DIR = Path("/home/bosho/FP/experiments/configs")

# Settings to add
SETTINGS = """
# Device configuration - all experiments default to GPU
device: cuda
keep_data_on_gpu: true
max_concurrent_clients: 10
"""

updated = 0
skipped = 0

for config_file in CONFIGS_DIR.rglob("*.yaml"):
    if config_file.name == "base.yaml":
        continue
    
    content = config_file.read_text()
    
    # Skip if already has device setting (exact match at start of line)
    if re.search(r'^device:\s*', content, re.MULTILINE):
        skipped += 1
        continue
    
    # Append settings to end of file
    config_file.write_text(content.rstrip() + "\n" + SETTINGS)
    updated += 1
    print(f"Updated: {config_file}")

print(f"\nUpdated: {updated} configs")
print(f"Skipped (already had device): {skipped} configs")
print(f"Total: {updated + skipped}")
