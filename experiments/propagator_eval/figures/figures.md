
## Main paper figures — highest priority

### Figure 1 — Intrinsic vs downstream decoupling

**Plot type:** scatter plot
**X-axis:** intrinsic MSE on Cora
**Y-axis:** downstream GCN accuracy on Cora
**Points:** APPNP, Chebyshev, Diffusion, Adjacency, Asym. RW
**Use:** Main methodological figure.

This is probably the **most important figure**. It shows that APPNP has the best intrinsic MSE, while Chebyshev gives the best downstream accuracy. That directly proves the paper’s central claim: downstream accuracy and intrinsic reconstruction measure different things.

Caption idea:

```latex
\caption{Intrinsic reconstruction quality does not fully predict downstream
accuracy on Cora. APPNP obtains the lowest feature-reconstruction MSE, while
Chebyshev achieves the highest downstream GCN accuracy.}
```

### Figure 2 — Quality–efficiency trade-off on OGBN-Arxiv

**Plot type:** scatter plot
**X-axis:** wall-clock time, log scale
**Y-axis:** intrinsic MSE
**Points:** APPNP, Diffusion, Chebyshev, Adjacency, Asym. RW
**Use:** Scale/practicality figure.

This shows that no operator dominates at scale. APPNP is fastest, Asym. RW has lowest MSE, and Diffusion/Chebyshev sit in the middle. This supports the practical guidance claim: operator choice is a trade-off, not a universal winner.

Caption idea:

```latex
\caption{Quality--efficiency trade-off on OGBN-Arxiv. APPNP is fastest but has
higher MSE, while asymmetric random walk gives the lowest MSE at substantially
higher cost.}
```

### Figure 3 — Downstream oracle-gap closure on Cora

**Plot type:** grouped bar chart
**X-axis:** operators
**Y-axis:** gap closed
**Groups:** GCN β=1, GCN β=10000, GAT β=1, GAT β=10000
**Use:** Strong downstream performance figure.

This shows Chebyshev is consistently strong downstream across backbones and beta regimes. Use this if you want the paper to look more practically compelling. If page space is tight, this can be a compact table instead.

Caption idea:

```latex
\caption{Downstream oracle-gap closure on Cora. Chebyshev closes the largest
fraction of the zero-hop to oracle gap across both GCN and GAT backbones.}
```

## Secondary figures — good for appendix or poster

### Figure 4 — Cora intrinsic MSE vs beta

**Plot type:** line plot
**X-axis:** β ∈ {1, 10, 10000}
**Y-axis:** MSE
**Lines:** APPNP, Chebyshev, Diffusion, Adjacency, Asym. RW
**Use:** Shows beta has negligible effect on intrinsic MSE in Cora.

This is useful because the report notes beta has little effect on intrinsic quality. That is an interesting negative result: partition label skew may not affect feature-level reconstruction as much as expected, at least on Cora.

### Figure 5 — Cora wall time vs beta

**Plot type:** line plot or grouped bar chart
**X-axis:** β
**Y-axis:** wall time
**Lines/bars:** operators
**Use:** Shows computational instability/cost differences.

This will make Chebyshev’s beta=10000 wall-time blow-up visible: 29.87s at β=1, 345.31s at β=10, 725.85s at β=10000. This is useful but maybe too implementation-specific for the main paper.

### Figure 6 — APPNP alpha sweep

**Plot type:** dual-axis line plot or two-panel plot
**X-axis:** α ∈ {0.05, 0.10, 0.20}
**Y-axis 1:** MSE / recovery ratio
**Y-axis 2:** iterations / wall time
**Use:** Shows teleportation strength trade-off.

This is a clean ablation. It shows higher α reduces MSE and convergence iterations, but also reduces cosine similarity. This is exactly the kind of operator-specific behavior the paper is about.

Best version: two panels.

```text
(a) α vs MSE / RR
(b) α vs Avg. Iters / Wall Time
```

### Figure 7 — Hop-depth L=1 vs L=2

**Plot type:** grouped bar chart
**X-axis:** operator × hop
**Y-axis:** MSE or RR
**Operators:** Adjacency, Diffusion
**Use:** Shows that L=2 improves reconstruction but costs more.

This is useful if reviewers ask about higher-order features. It shows the L=2 relaxation improves MSE and recovery ratio for both Adjacency and Diffusion, but adds topology/cost assumptions.

This is a good appendix figure, not main.

### Figure 8 — Diffusion tolerance sweep

**Plot type:** line plot
**X-axis:** ε ∈ {1e-2, 1e-3, 1e-4}
**Y-axis:** MSE / iterations
**Use:** Shows tolerance does not matter because diffusion hits iteration cap.

This is a diagnostic plot. It shows diffusion’s convergence issue is not just tolerance tuning. Useful for appendix or internal debugging, but not main-paper worthy.

## Downstream result figures

### Figure 9 — Best operator by dataset/backbone/beta

**Plot type:** heatmap
**Rows:** dataset × backbone × beta
**Columns:** operators
**Cell:** accuracy or gap closed
**Use:** Compact cross-dataset summary.

This is powerful for the poster. It shows Chebyshev wins many settings, but not all: Citeseer β=10000 favors Adjacency; Pubmed GAT β=1 favors Asym. RW; Pubmed GAT β=10000 favors Adjacency.

This figure supports the “no universal operator” claim.

### Figure 10 — Average downstream ranking across homophilic datasets

**Plot type:** horizontal bar chart
**X-axis:** average gap closed
**Y-axis:** operator
**Use:** Shows Chebyshev is best on average.

From your report:

```text
Chebyshev: 0.736
Adjacency: 0.715
Asym. RW: 0.714
APPNP: 0.695
Diffusion: 0.643
```

This is a clean summary. It may be better as a small table than a plot.

### Figure 11 — GCN vs GAT sensitivity to propagation

**Plot type:** grouped bar chart
**X-axis:** operator
**Y-axis:** gap closed
**Panels:** GCN and GAT
**Use:** Shows GAT starts much lower at zero-hop but recovers with propagation.

This is a nice story: propagation matters more visibly for GAT because zero-hop GAT collapses more severely on Cora.

## Heterophily figures

### Figure 12 — Heterophily gain over zero-hop

**Plot type:** bar chart
**X-axis:** operator
**Y-axis:** accuracy gain over zero-hop
**Panels:** Texas, Wisconsin
**Use:** Shows propagation is mixed in heterophily.

This should replace any claim like “APPNP dominates heterophily.” Your data says:

```text
Texas:
Adjacency +0.0432
Diffusion +0.0378
APPNP +0.0270

Wisconsin:
Diffusion +0.0157
APPNP ±0.0000
Adjacency -0.0313
```

This figure strongly supports the limitation claim: propagation can help, hurt, or be neutral in heterophilic graphs.

### Figure 13 — Heterophilic accuracy with error bars

**Plot type:** bar chart with std error bars
**X-axis:** operators
**Y-axis:** accuracy
**Panels:** Texas, Wisconsin
**Use:** Shows high variance due to small graph size.

Useful if you want to be transparent. But in the main paper, I would not spend space on it unless heterophily becomes a central claim.

## Intrinsic/process figures

### Figure 14 — Convergence rate by operator

**Plot type:** bar chart
**X-axis:** operator
**Y-axis:** convergence rate
**Datasets:** Cora and OGBN-Arxiv
**Use:** Shows APPNP converges reliably, while Diffusion/Chebyshev often hit iteration cap.

This is a good diagnostic figure because it supports the “process metrics matter” part of the protocol.

### Figure 15 — Wall time vs convergence rate

**Plot type:** scatter plot
**X-axis:** wall time
**Y-axis:** convergence rate
**Points:** operators
**Use:** Shows speed and convergence are not the same thing.

APPNP would look excellent here: fast and convergent. Asym. RW may be slow but not convergent at OGBN scale.

### Figure 16 — MSE vs cosine similarity

**Plot type:** scatter plot
**X-axis:** MSE
**Y-axis:** CosSim
**Use:** Shows squared-error recovery and directional alignment disagree.

This is a good intrinsic-methodology figure. On Cora, Chebyshev has best cosine similarity but not best MSE. That supports using multiple intrinsic metrics.

## Best final selection

For the **main CIKM short paper**, I recommend:

| Priority        | Figure/Table                                             | Why                                |
| --------------- | -------------------------------------------------------- | ---------------------------------- |
| Main Fig. 1     | Intrinsic MSE vs downstream accuracy on Cora             | Proves decoupling                  |
| Main Fig. 2     | OGBN quality–efficiency trade-off                        | Proves scale/practical trade-off   |
| Main Table 1    | Downstream gap-closure table across Cora/Citeseer/Pubmed | Shows practical performance        |
| Optional Fig. 3 | Heterophily gain over zero-hop                           | Shows limitation/regime dependence |

For the **poster / appendix / backup**, produce:

```text
Cora MSE vs beta
Cora wall time vs beta
APPNP alpha sweep
Hop-depth L=1 vs L=2
Diffusion epsilon sweep
Best-operator heatmap
Average downstream ranking
GCN vs GAT sensitivity
Heterophily accuracy with error bars
Convergence rate by operator
Wall time vs convergence rate
MSE vs cosine similarity
```

## IDE figure-generation checklist

When you produce these in the IDE:

```text
1. Export both PDF and PNG.
2. Use vector PDF for LaTeX.
3. Use large fonts: 8–9 pt minimum after scaling.
4. Avoid relying only on color; use markers/hatches if possible.
5. Use short captions that state the takeaway.
6. Add \Description{...} under every figure in ACM LaTeX.
7. Use consistent operator names: Adj, Diff, Cheb, APPNP, Asym. RW.
8. For log-scale plots, state it in axis label.
9. Do not include placeholder figures in final submission.
```

My strongest recommendation: **make Figure 1 first**. If that figure is clear, the paper’s novelty becomes clear.

[1]: https://cikm2026.diag.uniroma1.it/short-research-papers/?utm_source=chatgpt.com "Short Research Papers - CIKM 2026"
[2]: https://authors.acm.org/proceedings/production-information/describing-figures?utm_source=chatgpt.com "Describing Figures for ACM Publications"
