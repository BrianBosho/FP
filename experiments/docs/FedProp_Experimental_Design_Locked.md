**FedProp — Experimental Design**

**Locked Specification for Paper Experiments**

*From Literature Review to Committed Experimental Plan — Kipkirui & Barros, CMU Africa*

# **1\.  Executive Summary — What We Commit To**

This document translates the literature review into a committed experimental plan. It locks four claims, six primary results, two appendix result sets, a dataset list, a baseline list, a hyperparameter table, and a reporting format. Everything below is what we will run and report. Items marked as **CONDITIONAL** are subject to decision gates in §8.

## **The commit**

* **4 claims — A (zero-communication recovery), B (model-agnostic), C (partition robustness), D (heterophilic limitation).**

* **6 main-paper results — 2 tables (R1 GCN, R1b GAT), 4 figures (R2–R5), 1 heterophilic table (R6).**

* **4 appendix results — A1 Amazon, A2 PE ablation, A3 T sweep, A4 1-hop vs 2-hop.**

* **8 datasets — 3 main (Cora, Citeseer, Pubmed) \+ 1 conditional (OGBN-Arxiv) \+ 2 appendix (Amazon ×2) \+ 2 limitation (Texas, Wisconsin).**

* **6 primary baselines — Centralised, FedProp-Full, FedProp-Zero, FedProp (Adj), FedProp (Diff), FedGCN. FedGAT added for R1b.**

* **3 positioning baselines — FedSage+, FedDEP, FedCog — cited in related work, NOT in the main accuracy table.**

* **Total committed runs — approximately 1,980 training runs across all experiments (see §4 run matrix).**

| COMMITMENT STATEMENT This plan is the source of truth. Anything not listed here is out of scope for v1 of the paper. If new experiments are proposed, they must either displace something here or be explicitly added as a scope change. |
| :---- |

# **2\.  Claims and Supporting Experiments**

Every experiment must trace back to one of the four claims. If a proposed experiment does not support a claim, it does not belong in the paper.

| Claim | Statement | Experiments that support it | Where it appears in paper |
| :---: | ----- | ----- | ----- |
| **A** | Zero-additional-communication recovery of missing-neighbour performance. | R1 main accuracy table (Cora, Citeseer, Pubmed, OGBN-Arxiv)R2 accuracy-vs-communication plotR3 recovery / gap-closed plot | Table 1 (main accuracy). Figure 1 (communication trade-off). Figure 2 (gap closed). Abstract \+ Introduction leads. |
| **B** | Model-agnosticity (works with GCN and GAT backbones). | R1b GAT variant of main accuracy tableFedProp-GAT vs FedGAT comparison | Table 2 (GAT results in main paper). Method description (§4). Explicit backbone-agnostic claim in contributions. |
| **C** | Robustness across partition severity (IID ↔ non-IID) and client count. | R4 IID vs non-IID sweep (β \= 10000, 10, 1)R5 client-count scaling (K \= 1, 5, 10, 20\) | Figure 3 (β sweep). Figure 4 (K scaling). Discussion §5.3. |
| **D** | Homophily-dependent limitation on heterophilic graphs. | R6 Texas \+ Wisconsin heterophilic experimentsconnected to 3-part error bound (T1/T2/T3) | Table 3 (heterophilic results). Limitations section §6. Theory discussion §3.4. |

# **3\.  The Result Stack**

Six results for the main paper (R1–R6), four for the appendix (A1–A4). Each is scoped to a single clear purpose.

| ID | Title | Datasets | Methods compared | Axes / what to report | Paper placement |
| :---: | ----- | ----- | ----- | ----- | ----- |
| **R1** | **Core accuracy table (GCN)** | Cora, Citeseer, Pubmed+ OGBN-Arxiv (conditional) | Centralised GCN, FedProp-Full, FedProp-Zero,FedProp (Adj), FedProp (Diff),FedGCN (1-hop) | Mean ± std test accuracy over 10 seeds.β ∈ {10000, 10, 1} in three sub-rows. | MAIN paperTable 1 |
| **R1b** | **GAT variant accuracy table** | Cora, Citeseer, Pubmed | Centralised GAT, FedProp-GAT-Full/Zero/Adj/Diff, FedGAT | Same format as R1. 10 seeds. β ∈ {10000, 10, 1}. | MAIN paperTable 2 |
| **R2** | **Accuracy vs. communication** | Cora \+ OGBN-Arxiv | FedProp (Diff), FedGCN (0/1/2-hop), FedGAT.(FedCog estimate if available.) | X \= total bytes transferred per client over training  (pre-train \+ Σ rounds × model params).Y \= final test accuracy.One point per method; Pareto frontier. | MAIN paperFigure 1 |
| **R3** | **Recovery / gap-closed bar plot** | Cora, Citeseer, Pubmed | FedProp-Zero, FedProp (best variant), FedProp-Full, Centralised | Bar height \= test accuracy.Annotate % of gap closed (Full − Zero). | MAIN paperFigure 2 |
| **R4** | **IID ↔ non-IID sweep** | Cora | FedProp (Diff), FedProp-Zero, FedProp-Full, FedGCN | X \= β ∈ {10000, 100, 10, 1}.Y \= test accuracy.Line plot, 10 seeds per point. | MAIN paperFigure 3 |
| **R5** | **Client-count scaling** | Cora | FedProp (Diff), FedProp-Zero, FedGCN | X \= K ∈ {1, 5, 10, 20}.Y \= test accuracy.Three lines per method (β=10000, β=10, β=1). | MAIN paperFigure 4 |
| **R6** | **Heterophilic limitation** | Texas, Wisconsin | Centralised GCN, FedProp-Full/Zero,FedProp (Diff), FedGCN | Accuracy table (mean ± std, 10 seeds).Homophily ratio reported. Discussed via T1/T2/T3 error bound. | MAIN paperTable 3 \+Limitations § |
| **A1** | **Amazon extended validation** | Amazon-Computers, Amazon-Photos | All R1 methods | Same format as R1. | APPENDIX |
| **A2** | **Positional encoding ablation** | Cora, Citeseer, Pubmed | FedProp (Adj, Adj+PE, Diff, Diff+PE) | Compare ± PE effect. 10 seeds. | APPENDIX |
| **A3** | **Propagation depth sensitivity** | Cora, Pubmed | FedProp (Diff), T ∈ {10, 25, 50, 100} | Line plot; find saturation T. | APPENDIX |
| **A4** | **1-hop vs 2-hop accessibility** | Cora, Citeseer | FedProp L=1, L=2 variants | Accuracy comparison \+ privacy discussion. | APPENDIX |

# **4\.  Run Matrix — Total Training Runs**

Every run counted. Many runs are reused across results (for example, R2's communication plot uses the same numbers as R1's accuracy table). The matrix below counts only the training runs that must actually be executed.

| Result | Experiment description | Datasets | Methods | Seeds | Settings | Total runs | Status |
| ----- | ----- | ----- | ----- | ----- | ----- | :---: | ----- |
| **R1** | Core accuracy (GCN): 1-hop \+ 2-hop | 3 | 6 | 10 | 3 (β) | 3×6×10×3 \= 540 | Cora/Cite/Pub: need to run. |
| **R1 (OGBN)** | Core accuracy (GCN) on OGBN-Arxiv | 1 | 6 | 5 | 3 (β) | 1×6×5×3 \= 90 | Conditional on gate (§ below) |
| **R1b** | Core accuracy (GAT): 1-hop \+ 2-hop | 3 | 6 | 10 | 3 (β) | 3×6×10×3 \= 540 | Need to run |
| **R2** | Accuracy-vs-comm. plot (re-uses R1 runs) | 2 | 5 | — | — | 0 new runs | Post-hoc from R1 |
| **R3** | Recovery bar plot (re-uses R1 runs) | 3 | 4 | — | — | 0 new runs | Post-hoc from R1 |
| **R4** | β sweep: 10000, 100, 10, 1 | 1 (Cora) | 4 | 10 | 4 (β) | 1×4×10×4 \= 160 | Need to run |
| **R5** | K scaling: 1, 5, 10, 20 clients | 1 (Cora) | 3 | 10 | 4 (K) × 3 (β) | 1×3×10×12 \= 360 | Partial; some from R1 already |
| **R6** | Heterophilic | 2 | 5 | 10 | 3 (β) | 2×5×10×3 \= 300 | Need to run |
| **A1** | Amazon extended | 2 | 6 | 10 | 3 (β) | 2×6×10×3 \= 360 | Mostly done; write-up needed |
| **A2** | PE ablation | 3 | 4 | 10 | 3 (β) | 3×4×10×3 \= 360 | Have results; organise |
| **A3** | Prop. depth T sweep | 2 | 1 | 5 | 4 (T) | 2×1×5×4 \= 40 | Need to run |
| **A4** | 1-hop vs 2-hop | 2 | 2 | 10 | 3 (β) | 2×2×10×3 \= 120 | Need to run |
|  | **TOTAL** |  |  |  |  | **≈ 2,770 training runs** |  |

**Note on R1/R1b:** The above counts the standard FedProp variants per dataset per hop. With 2 hop variants (L=1, L=2) × 3 variants per config (standard, centralised, fedprop_full), actual atomic runs are: R1 = 1,260 (citations) + 180 (OGBN) = 1,440; R1b = 900. Full Track A matrix = ≈ 3,930 runs.

| Compute budget Compute budget estimate: Cora/Citeseer/Pubmed runs take minutes each on a modern GPU. OGBN-Arxiv runs take approximately 20–30 minutes each. Heterophilic runs take under 2 minutes each. Rough total wall time with a single GPU: approximately 3–4 days of compute to finish the entire matrix from scratch. Many of the R1 and A1 runs are already complete or partially complete. |
| :---- |

# **5\.  Hyperparameter Lock**

Final committed hyperparameters per dataset and backbone. Every choice is traceable to a paper cited in the literature review. No further tuning — if hyperparameters need to change during execution, this table updates and we re-run affected results.

| Dataset | Backbone | Layers | Hidden | Dropout | Optim | LR | WD | Epochs / ES | Justification (source) |
| ----- | ----- | :---: | :---: | :---: | :---: | :---: | :---: | ----- | ----- |
| **Cora** | GCN | 2 | 16 | 0.5 | SGD | 0.5 | 5e-4 | 600 max \+ ES patience 10 | Kipf & Welling architecture; SGD/lr=0.5 from original codebase |
| **Cora** | GAT | 2 | 64 (8×8) | 0.6 | Adam | 0.005 | 5e-4 | 600 max \+ ES patience 10 | Velickovic et al. 2018 exact setting |
| **Citeseer** | GCN | 2 | 16 | 0.5 | SGD | 0.5 | 5e-4 | 600 max \+ ES patience 10 | Same as Cora (consistent w/ Planetoid family) |
| **Citeseer** | GAT | 2 | 64 (8×8) | 0.6 | Adam | 0.005 | 5e-4 | 600 max \+ ES patience 10 | Velickovic et al. 2018 |
| **Pubmed** | GCN | 2 | 16 | 0.5 | Adam | 0.01 | 5e-4 | 600 max \+ ES patience 10 | Kipf & Welling defaults; Adam works better on Pubmed scale |
| **Pubmed** | GAT | 2 | 64 (8×8) | 0.6 | Adam | 0.01 | 5e-4 | 600 max \+ ES patience 10 | Velickovic et al. 2018 (higher LR for Pubmed) |
| **OGBN-Arxiv** | GCN | 3 | 256 | 0.5 | Adam | 0.01 | 0 | 500 max \+ ES patience 20 | OGB baseline (Hu et al. 2020); adds BatchNorm between layers |
| **Amazon-CS** | GCN | 2 | 64 | 0.5 | Adam | 0.01 | 5e-4 | 600 max \+ ES patience 10 | Community standard for dense co-purchase graphs |
| **Amazon-Photo** | GCN | 2 | 64 | 0.5 | Adam | 0.01 | 5e-4 | 600 max \+ ES patience 10 | Consistent with Amazon-CS |
| **Texas** | GCN | 2 | 32 | 0.5 | Adam | 0.01 | 5e-4 | 400 max \+ ES patience 20 | Pei et al. 2020 (Geom-GCN) reference; small graph → moderate hidden |
| **Wisconsin** | GCN | 2 | 32 | 0.5 | Adam | 0.01 | 5e-4 | 400 max \+ ES patience 20 | Same protocol as Texas |

## **Federated protocol (applies to all datasets)**

* **Number of clients: K \= 10 (primary); R5 additionally sweeps K ∈ {1, 5, 10, 20} on Cora only.**

* **Aggregation: FedAvg, synchronous, weighted by client data size.**

* **Global rounds: 200 primary \+ early stopping on validation loss, patience 10 (consistent across all experiments).**

* **Local epochs per round: 1 (standard FedAvg).**

* **Partition: Dirichlet label distribution. β ∈ {1, 10, 10000} across all R1/R1b/R5/R6/A1/A2/A4 results. R4 additionally sweeps β=100 as an intermediate point.**

* **FedProp propagation: T \= 50 iterations. Diffusion step size h \= 0.1. Boundary reset after each local iteration.**

* **Seeds: 10 random seeds per configuration. Mean ± standard deviation reported.**

# **6\.  Dataset Lock**

Eight datasets total, in three categories: MAIN paper (4), APPENDIX (2), LIMITATION (2). No other datasets will be added without changing this document.

| Dataset | Role | |V|/|E| | Split protocol | Partition protocol | Justification |
| ----- | :---: | :---: | ----- | ----- | ----- |
| **Cora** | **MAIN** | 2,708 / 5,429 | Planetoid fixed(20/class train, 500 val, 1000 test) | Dirichlet β=10000, β=10, β=1 | Standard benchmark. Matches FedGCN/FedGAT protocol exactly. Reviewers expect it. |
| **Citeseer** | **MAIN** | 3,327 / 4,732 | Planetoid fixed(120 train, 500 val, 1000 test) | Dirichlet β=10000, β=10, β=1 | Sparse adjacency stress test (avg deg 2.84 is smallest in set). Standard. |
| **Pubmed** | **MAIN** | 19,717 / 44,338 | Planetoid fixed(60 train, 500 val, 1000 test) | Dirichlet β=10000, β=10, β=1 | Larger-scale citation. Standard. Matches all primary baselines. |
| **OGBN-Arxiv** | **MAIN(conditional)** | 169,343 / 1.17M | OGB standard(\~54% / 18% / 28%) | Dirichlet β=10000, β=10, β=1 | Scale differentiator vs FedSage+/FedCog. Only FedGCN tests here. Gate conditions below (§8). |
| **Amazon-Computers** | **APPENDIX** | 13,752 / 287K | Random 60/20/20(10 seeds) | Dirichlet β=10000, β=10, β=1 | Extended validation beyond citation graphs. |
| **Amazon-Photos** | **APPENDIX** | 7,650 / 238K | Random 60/20/20(10 seeds) | Dirichlet β=10000, β=10, β=1 | Dense graph test (avg deg 62). |
| **Texas** | **LIMITATION** | 183 / 325 | Geom-GCN random60/20/20 (10 seeds) | Dirichlet β=10000, β=10 | Heterophilic (h=0.11). Reviewer-required. Small → runs in minutes. |
| **Wisconsin** | **LIMITATION** | 251 / 499 | Geom-GCN random60/20/20 (10 seeds) | Dirichlet β=10000, β=10 | Heterophilic (h=0.20). Reviewer-required. Small → runs in minutes. |

# **7\.  Baseline Lock**

Eleven baseline/method entries. Six are primary (appear in main accuracy tables). Three are positioning only (cited in related work, not in numerical comparison tables). Two are upper and lower bounds.

| Method | Role | What it represents | Comparison protocol | Where it appears |
| ----- | :---: | ----- | ----- | ----- |
| **Centralised GCN/GAT** | **Upper bound** | Full graph access, no federation. The performance ceiling we target. | Same architecture and hyperparameters as federated variants. | R1, R1b, R3, R6; referenced in abstract. |
| **FedProp-Full (oracle)** | **Upper bound** | Federated but with exact 1-hop remote features available. | Same FedAvg \+ 10 clients \+ same HPs. | R1, R1b, R3, R6. |
| **FedProp-Zero** | **Lower bound** | Federated with no imputation. Remote neighbours are featureless. | Same FedAvg \+ 10 clients \+ same HPs. | R1, R1b, R3, R4, R5, R6. |
| **FedProp (Adj)** | **Ablation** | Our method with normalised-adjacency propagation (simpler variant). | T=50 iterations; no PE. | R1, R1b, A2 (with PE). |
| **FedProp (Diff)** | **OUR METHOD** | Our primary method. Discrete diffusion / heat-equation propagation. | T=50 iterations; h=0.1; no PE in main table. \+PE variant in A2. | R1, R1b, R2, R3, R4, R5, R6. |
| **FedGCN (1-hop)** | **Main competitor** | Transmission-based baseline. Shares 1-hop features in pre-training. | Same HPs; report pre-training communication cost. | R1, R2, R4, R5, R6. |
| **FedGCN (2-hop)** | **Main competitor** | As above but with 2-hop feature exchange. | Same HPs. | R2 only (communication trade-off). |
| **FedGAT** | **Main competitor** | GAT-specific transmission method. Chebyshev approximation. | Degree-16 Chebyshev; same partition protocol. | R1b only (GAT backbone). |
| **FedSage+** | **Positioning(not in main table)** | Generative baseline. Different protocol family (Louvain \+ 60/20/20). | Cited in related work. NOT in main accuracy table. | Related work §2. Limitations / cross-family paragraph §5.5. |
| **FedDEP** | **Positioning** | Same family as FedSage+. Deeper generator. | Cited in related work. | Related work §2. |
| **FedCog** | **Positioning** | Border-embedding sharing method. Conceptually close — flagged by reviewers. | Cited; dedicated contrast paragraph. | Related work §2. Novelty comparison paragraph §2.3. |

| Cross-family disclaimer FedSage+, FedDEP, and FedCog will NOT appear in Table 1, Table 2, or Table 3 as numerical competitors. They belong to a different protocol family (Louvain \+ inductive 60/20/20). Instead: FedSage+ and FedDEP are discussed as a cross-family positioning paragraph in §5.5. FedCog gets a dedicated novelty-contrast paragraph in §2.3 (reviewer-flagged as similar). If a reviewer specifically requests FedSage+ numbers, the response is: 'Under the original FedSage+ protocol, results are reported in Appendix X; under our Dirichlet protocol, FedSage+ is out of scope because its generator training is not compatible with transductive fixed-split evaluation.' |
| :---- |

# **8\.  Decision Gates and Risk Management**

Three open decisions with hard gates. Each has a trigger condition and a fallback.

## **Gate 1 — OGBN-Arxiv inclusion in main paper**

**Trigger:** FedProp (Diff) on OGBN-Arxiv completes with stable 10-seed results within 7 days of starting the run.

**If met:** Include OGBN-Arxiv as the 4th dataset in Table 1\. Add to Figure 1 as the second subpanel.

**If not met:** Move OGBN-Arxiv to an appendix section 'Scale discussion' with the runs we have. Note as future work. Keep the main tables at 3 datasets. Do not let this delay submission.

## **Gate 2 — GAT main table vs appendix**

**Decision:** GAT (Table 2\) goes in the MAIN paper. This is the cleanest way to support Claim B (model-agnosticity) and also directly answers reviewers asking for FedProp-GAT vs FedGAT.

**Fallback:** If page count forces compression, the GAT table can move to the appendix BUT the claim of backbone-agnosticity must then be downgraded from 'shown' to 'demonstrated in supplementary.'

## **Gate 3 — Heterophilic results are worse than expected**

**Expected outcome:** FedProp degrades on Texas/Wisconsin — accuracy drops relative to FedProp-Full. This is the claim of the section.

**If FedProp actually matches FedProp-Full on heterophilic:** This would be a surprise and a stronger result. Reframe the discussion around why propagation survives heterophily (likely because of the PE regularizer or the boundary reset). Update Claim D to 'limited degradation.'

**If FedProp crashes / catastrophically fails:** Investigate numerically. Possible cause: diffusion step size h too aggressive. Try h ∈ {0.05, 0.1, 0.2}. Report whichever h gives stable results, and note the sensitivity.

# **9\.  Reporting Format — What Goes in the Paper**

Every table and figure is pre-specified. No surprises during writing. Each item below maps to an experiment from §3.

| Paper element | Type | Content | Format | Source result(s) |
| ----- | :---: | ----- | ----- | ----- |
| **Table 1** | Table | Core GCN accuracy: Cora, Citeseer, Pubmed, OGBN-Arxiv × IID/non-IID × 6 methods | Mean ± std (3 decimal places). Bold best per column. Highlight FedProp (Diff). | R1 \+ R1 (OGBN) |
| **Table 2** | Table | Core GAT accuracy: Cora, Citeseer, Pubmed × IID/non-IID × 6 methods | Same format as Table 1\. | R1b |
| **Table 3** | Table | Heterophilic: Texas \+ Wisconsin × IID/non-IID × 5 methods | Same format; add edge-homophily ratio as a header row. | R6 |
| **Figure 1** | Figure | Accuracy (%) vs. total communication bytes (log scale) | Scatter plot; one colour per method; Cora \+ OGBN-Arxiv as subpanels. | R2 |
| **Figure 2** | Figure | Recovery bar plot: Zero → FedProp(Diff) → Full → Centralised | Grouped bars per dataset with % gap closed annotation. | R3 |
| **Figure 3** | Figure | Accuracy vs. β (Dirichlet partition severity) | Line plot on log-β axis; 4 methods; error bands \= std over 10 seeds. | R4 |
| **Figure 4** | Figure | Accuracy vs. number of clients K | Line plot K ∈ {1,5,10,20}; 3 methods; two panels (IID, non-IID). | R5 |
| **Appx Table A1** | Table | Amazon-Computers \+ Amazon-Photos | Same format as Table 1\. | A1 |
| **Appx Table A2** | Table | Positional encoding effect (Adj, Adj+PE, Diff, Diff+PE) | Delta from no-PE baseline as a separate column. | A2 |
| **Appx Figure A3** | Figure | T sweep: accuracy vs. propagation iterations | Line plot; T ∈ {10, 25, 50, 100}. | A3 |
| **Appx Table A4** | Table | Accessible radius comparison (L=1 vs L=2) | Discuss privacy implications alongside accuracy. | A4 |

## **Formatting conventions**

* **Accuracy reported as percentage with 1 decimal place (e.g., 80.8% ± 1.4) in the main text; 3 decimals (0.808 ± 0.014) in tables.**

* **Bold the best method per column. Highlight FedProp rows with a subtle background shade if possible.**

* **Communication cost reported in MB per training session, broken down as 'pre-training' \+ 'per-round × rounds' \= total.**

* **Every table and figure caption must state: dataset, partition (β), number of clients, number of seeds.**

* **Report gap closed \= (FedProp − Zero) / (Full − Zero) × 100% in the recovery discussion.**

# **10\.  Execution Priority**

The order in which remaining experiments should be run. Highest priority at top.

## **Week 1 — Critical path**

* **1\. Finalise R1 on Cora, Citeseer, Pubmed (10 seeds × IID/non-IID × all 6 methods). Lock numbers.**

* **2\. Start R1 OGBN-Arxiv runs. Kick off early because of Gate 1 timing.**

* **3\. Complete R1b GAT runs (any missing cells).**

## **Week 2 — Figures and extended**

* **4\. Run R4 β sweep (160 runs on Cora only — fast).**

* **5\. Run R5 K scaling (240 runs on Cora only — fast).**

* **6\. Produce Figures 1–4 from R1, R2, R3, R4, R5.**

* **7\. Finalise A1 Amazon write-up (runs already done).**

## **Week 3 — Limitation \+ appendix**

* **8\. Run R6 heterophilic (200 runs, fast — under 1 day total).**

* **9\. Run A3 T sweep \+ A4 1-hop/2-hop.**

* **10\. Organise A2 PE ablation from existing results.**

* **11\. Draft experimental section of paper using all locked numbers.**

*End of Locked Experimental Design — FedProp / Kipkirui & Barros / Carnegie Mellon University Africa*