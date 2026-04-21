# Research-Production Refactor Plan

**Date:** 2026-04-21
**Scope:** Remaining documentation, package refactor, analysis workflow, and
verification work after closing the FL performance checklist.

## Status

The FL correctness and performance checklist is complete:

- `docs/FL_PERFORMANCE_CHECKLIST.md` has no remaining `[ ]` or `[~]` items.
- The completed performance work is committed through
  `7149eba Phase final: close remaining FL checklist items`.
- `conf/publication.yaml` now provides safer defaults for new paper runs.

The remaining work is not primarily FL algorithm correctness. It is codebase
organization, package boundaries, analysis workflow, and doc consistency so the
repository is easier to use for publication experiments.

## Goal

Finish the remaining `/docs` roadmap items so the repository has:

- truthful, non-contradictory documentation,
- a clear `src/fedgnn` package as the source of truth,
- legacy `src.*` entrypoints that continue to work,
- reusable analysis utilities instead of notebook copy-paste,
- tests that protect the package structure and CLI behavior.

## Phase 0: Reconcile Documentation

**Files:**

- `docs/FL_FIX_IMPLEMENTATION_PLAN.md`
- `docs/FL_IMPLEMENTATION_REVIEW.md`
- `docs/FL_PERFORMANCE_CHECKLIST.md`
- `docs/CODEBASE_REFACTOR_PLAN.md`

**Actions:**

1. Remove or rewrite stale language in `FL_FIX_IMPLEMENTATION_PLAN.md` that
   still describes completed FL blockers as open.
2. Make `FL_IMPLEMENTATION_REVIEW.md` explicitly historical context.
3. Keep `FL_PERFORMANCE_CHECKLIST.md` closed and stop treating it as the active
   roadmap.
4. Make `CODEBASE_REFACTOR_PLAN.md` the active tracker for remaining package and
   analysis refactor work.
5. Link this plan from the relevant docs once the team adopts it.

**Done When:**

- No docs say completed FL performance blockers are still open.
- `CODEBASE_REFACTOR_PLAN.md` clearly owns the remaining work.
- A search for open checklist markers in FL docs does not surface stale items.

**Suggested Commit:**

```bash
git commit -m "docs: reconcile FL review and refactor roadmap"
```

## Phase 1: Finish Package Refactor

**Active tracker:** `docs/CODEBASE_REFACTOR_PLAN.md`, Phase C / C4.

The desired end state is:

- `src/fedgnn/...` contains the real implementation.
- Legacy `src.*` modules remain as compatibility wrappers.
- Existing CLIs and imports keep working during and after migration.

### Phase 1A: Migrate Utilities

**Move source of truth to:**

- `src/fedgnn/utils`

**Keep wrappers in:**

- `src/utils`

**Actions:**

1. Move reusable path, config, logging, and result helpers into
   `src/fedgnn/utils`.
2. Replace old utility modules with thin wrappers that import from
   `src.fedgnn.utils`.
3. Update internal imports to prefer the new package path.

**Verification:**

```bash
python3 -m compileall -q src
python3 -m pytest -q
```

**Suggested Commit:**

```bash
git commit -m "refactor: migrate utils into fedgnn package"
```

### Phase 1B: Migrate Data Modules

**Move source of truth to:**

- `src/fedgnn/data`

**Keep wrappers in:**

- `src/dataprocessing`

**Actions:**

1. Move graph loading, partitioning, propagation, positional encoding, and
   dataset helpers into `src/fedgnn/data`.
2. Keep old `src/dataprocessing.*` imports working through wrappers.
3. Update internal imports to use `src.fedgnn.data`.
4. Verify publication config paths still run.

**Verification:**

```bash
python3 -m compileall -q src
python3 -m pytest -q
```

**Suggested Commit:**

```bash
git commit -m "refactor: migrate data modules into fedgnn package"
```

### Phase 1C: Migrate Models

**Move source of truth to:**

- `src/fedgnn/models`

**Keep wrappers in:**

- `src/models.py`

**Actions:**

1. Move model definitions into `src/fedgnn/models`.
2. Keep `src.models` as a compatibility import layer.
3. Update train/server/runtime imports to prefer `src.fedgnn.models`.

**Verification:**

```bash
python3 -m compileall -q src
python3 -m pytest -q
```

**Suggested Commit:**

```bash
git commit -m "refactor: migrate models into fedgnn package"
```

### Phase 1D: Migrate FL Runtime

**Move source of truth to:**

- `src/fedgnn/fl`

**Keep wrappers in:**

- `src/client.py`
- `src/server.py`
- `src/train.py`
- `src/run.py`

**Actions:**

1. Move client, server, training, and orchestration logic into `src/fedgnn/fl`.
2. Keep legacy module entrypoints and imports working.
3. Preserve Ray actor behavior and config-driven GPU reservation.
4. Preserve completed FL fixes: FedAvg, FedBN, seed plumbing, global eval
   preprocessing, and training knobs.

**Verification:**

```bash
python3 -m compileall -q src
python3 -m pytest -q
python3 -m src.run --help
```

**Suggested Commit:**

```bash
git commit -m "refactor: migrate FL runtime into fedgnn package"
```

### Phase 1E: Migrate Experiment Runner

**Move source of truth to:**

- `src/fedgnn/experiments`

**Keep wrappers in:**

- `src/experiments`

**Actions:**

1. Move reusable experiment runner logic into `src/fedgnn/experiments`.
2. Keep `python -m src.experiments.run_experiments ...` working.
3. Preserve benchmark harness behavior and subprocess-per-seed assumptions.

**Verification:**

```bash
python3 -m compileall -q src
python3 -m pytest -q
python3 -m src.experiments.run_experiments --help
```

**Suggested Commit:**

```bash
git commit -m "refactor: migrate experiment runner into fedgnn package"
```

## Phase 2: Add Package Invariant Tests

**Active tracker:** `docs/CODEBASE_REFACTOR_PLAN.md`, Phase C / C5.

**Actions:**

1. Add import tests for the new package:
   - `src.fedgnn`
   - `src.fedgnn.data`
   - `src.fedgnn.fl`
   - `src.fedgnn.models`
   - `src.fedgnn.utils`
2. Add compatibility import tests for legacy `src.*` modules.
3. Add CLI smoke tests that do not require datasets or heavy dependencies:
   - `python3 -m src.run --help`
   - `python3 -m src.experiments.run_experiments --help`
4. Add a wrapper-direction test after migration:
   - legacy modules import from `src.fedgnn`,
   - `src.fedgnn` modules do not import implementation from legacy modules.

**Done When:**

- `C5` can be marked `[x]`.
- Phase C acceptance criteria can be marked `[x]`.
- Tests fail if future changes accidentally make legacy modules the source of
  truth again.

**Suggested Commit:**

```bash
git commit -m "test: add package refactor invariants"
```

## Phase 3: Consolidate Analysis and Notebook Workflow

**Active tracker:** `docs/CODEBASE_REFACTOR_PLAN.md`, Phase D.

**Target package:**

- `src/fedgnn/analysis`

**Actions:**

1. Inspect notebooks and benchmark scripts for duplicated parsing, plotting, and
   table-building logic.
2. Move reusable logic into modules, for example:
   - `src/fedgnn/analysis/results.py`
   - `src/fedgnn/analysis/plots.py`
   - `src/fedgnn/analysis/tables.py`
3. Keep notebooks thin:
   - load result paths,
   - call reusable functions,
   - render figures and tables.
4. Confirm notebook-generated files go to ignored output locations such as
   `runs/`, unless explicitly curated.
5. Decide how to treat `reports/`:
   - keep it for curated artifacts and document what belongs there, or
   - mark the optional export path as not adopted.

**Done When:**

- `D2` is marked `[x]`.
- `D3` is either marked `[x]` or explicitly documented as not adopted.
- Notebooks run without generating tracked junk.
- Analysis logic is mostly importable Python, not notebook copy-paste.

**Suggested Commit:**

```bash
git commit -m "refactor: consolidate analysis utilities"
```

## Phase 4: Final Verification and Checklist Closure

**Actions:**

1. Run syntax and tests:

   ```bash
   python3 -m compileall -q src
   python3 -m pytest -q
   git diff --check
   ```

2. Check docs for remaining open checklist markers:

   ```bash
   rg -n "^- \[[ ~]\]|^- \[~\]" docs
   ```

3. Check legacy entrypoints:

   ```bash
   python3 -m src.run --help
   python3 -m src.experiments.run_experiments --help
   ```

4. If dependencies and data are available, run a small publication-config smoke
   experiment:

   ```bash
   python3 -m src.experiments.run_experiments --config conf/publication.yaml
   ```

5. Update `docs/CODEBASE_REFACTOR_PLAN.md` so completed items are marked `[x]`
   and any consciously deferred optional items are clearly labeled.

**Done When:**

- No unintended open `[ ]` or `[~]` items remain in `/docs`.
- Tests pass.
- Legacy entrypoints still work.
- `src/fedgnn` is the source of truth.
- `git status` is clean except intentional local files such as `datasets`.

**Suggested Commit:**

```bash
git commit -m "docs: close codebase refactor checklist"
```

## Recommended Execution Order

1. Phase 0: reconcile docs.
2. Phase 1A: migrate utilities.
3. Phase 1B: migrate data modules.
4. Phase 1C: migrate models.
5. Phase 1D: migrate FL runtime.
6. Phase 1E: migrate experiment runner.
7. Phase 2: add package invariant tests.
8. Phase 3: consolidate analysis utilities.
9. Phase 4: final verification and close the refactor checklist.

This order keeps risk controlled: make the docs truthful first, migrate one
boundary at a time, and then lock the package structure with tests.
