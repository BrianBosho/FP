# R1b Experiment Results — GAT Accuracy Table

**Model:** GAT | **Clients:** 10 | **Rounds:** 400 | **Reps:** 10 | **Optimizer:** SGD lr=0.5 (Cora/Citeseer), Adam lr=0.01 (Pubmed)
**FP settings:** num_iterations=50, diffusion_t=0.1 | **hop:** 2 for all conditions
*Last updated: 2026-04-30 13:10*

---

## Cora

### PE (use_pe=true)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| adjacency   | 0.7754±0.0068 | 0.7839±0.0087 | 0.8178±0.0086 | ✅ |
| diffusion   | 0.7563±0.0082 | 0.7533±0.0092 | 0.7990±0.0062 | ✅ |
| full        | 0.8131±0.0069 | 0.8091±0.0070 | 0.8069±0.0052 | ✅ |

### No PE (use_pe=false)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| zero_hop    | 0.6162±0.0038 | 0.6145±0.0066 | 0.6717±0.0072 | ✅ |
| adjacency   | 0.7453±0.0224 | 0.7483±0.0144 | 0.8018±0.0075 | ✅ |
| diffusion   | 0.7014±0.0089 | 0.7017±0.0196 | 0.7688±0.0111 | ✅ |
| full        | 0.8105±0.0086 | 0.8098±0.0091 | 0.8138±0.0063 | ✅ |

---

## Citeseer

### PE (use_pe=true)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| adjacency   | 0.6635±0.0082 | 0.6588±0.0064 | 0.6910±0.0041 | ✅ |
| diffusion   | 0.6509±0.0069 | 0.6457±0.0084 | 0.6865±0.0070 | ✅ |
| full        | 0.6827±0.0093 | 0.6851±0.0136 | 0.6931±0.0067 | ✅ |

### No PE (use_pe=false)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| zero_hop    | 0.5865±0.0063 | 0.5650±0.0074 | 0.6004±0.0082 | ✅ |
| adjacency   | 0.6550±0.0169 | 0.6481±0.0184 | 0.6800±0.0101 | ✅ |
| diffusion   | 0.6178±0.0179 | 0.6099±0.0122 | 0.6378±0.0178 | ✅ |
| full        | 0.6832±0.0090 | 0.6898±0.0089 | 0.6914±0.0099 | ✅ |

---

## Pubmed

### PE (use_pe=true)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| adjacency   | 0.7875±0.0067 | 0.7810±0.0089 | 0.7518±0.0039 | ✅ |
| diffusion   | 0.7861±0.0046 | 0.7812±0.0054 | 0.7425±0.0112 | ✅ |
| full        | N/A | N/A | N/A | — PE not applied with full loading |

### No PE (use_pe=false)

| Data Loading | Beta 10000 | Beta 10 | Beta 1 | Done |
|-------------|------------|---------|--------|------|
| zero_hop    | 0.6430±0.0091 | 0.6759±0.0047 | 0.6557±0.0165 | ✅ |
| adjacency   | 0.7933±0.0061 | 0.7756±0.0088 | 0.7261±0.0047 | ✅ |
| diffusion   | 0.7555±0.0121 | 0.7522±0.0098 | 0.7112±0.0253 | ✅ |
| full        | 0.7866±0.0040 | 0.7874±0.0123 | 0.7194±0.0083 | ✅ |

---

> ✅ = done  — = not applicable (PE not supported for this condition)
