# Experiment Config Layout

Configs are organized by experiment intent first, then by dataset family:

```text
experiments/configs/
├── shared/                  # Shared defaults and seed schedules.
├── planetoid/               # Cora, Citeseer, Pubmed.
├── large/                   # ogbn-arxiv, Amazon Computers, Amazon Photos.
├── heterophilic/            # Texas, Wisconsin, Actor, Roman-empire, etc.
├── ablations/               # Cross-dataset sweeps and diagnostic studies.
├── reruns/                  # Top-ups and failed-row reruns.
├── smoke/                   # Fast sanity checks.
└── archive/                 # Historical configs kept for provenance.
```

Within dataset families, use:

```text
<family>/<dataset>/<backbone>/<hop>/
```

For example:

```text
planetoid/cora/gcn/hop1/main.yaml
planetoid/cora/gat/hop2/pe.yaml
large/amazon-computers/gcn/hop2/main.yaml
heterophilic/roman-empire/gcn/hop1/main.yaml
```

## File Naming

- `main.yaml`: canonical table run for that dataset/backbone/hop.
- `pe.yaml`: positional-encoding variant.
- `topups.yaml`: seed completion for incomplete rows.
- `quickval.yaml`: short validation run.
- `preflight.yaml`: cheap config/runtime validation.
- `client_sweep.yaml`: client-count scalability sweep.
- Use explicit suffixes only when a directory needs multiple focused variants, such as `pe_adjacency.yaml`.

## Dataset Families

Planetoid:

- `cora`
- `citeseer`
- `pubmed`

Large:

- `ogbn-arxiv`
- `amazon-computers`
- `amazon-photos`

Heterophilic:

- `texas`
- `wisconsin`
- `actor`
- `roman-empire`
- `amazon-ratings`
- `minesweeper`

## Migration Map

- `R1/` moved to `planetoid/*/gcn/hop1/` and `large/ogbn-arxiv/gcn/hop1/`.
- `R1b/` moved to `planetoid/*/gat/hop2/`; notes and launch scripts moved to `archive/R1b/`.
- `R5/` moved to `ablations/client_count/`.
- `R6/` moved to `heterophilic/texas/gcn/hop1/` and `heterophilic/wisconsin/gcn/hop1/`.
- `R7/` moved to `large/amazon-computers/gcn/` and `large/amazon-photos/gcn/`.
- `A2/` moved to `ablations/pe/`.
- `A3/` moved to `ablations/propagation_operator/`.
- `A4/` moved to `ablations/hop_depth/`.
- `rerun/` moved to `reruns/topups/` and `reruns/failed_rows/`.
- Historical baseline, root test, generated scalability, and older ablation configs moved under `archive/`.

Runner code may still reference old paths until it is updated. Treat `archive/` as provenance, not the place for new experiments.
