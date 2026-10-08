# AAAI-2027 Adversarial Quality Audit

Date: 2026-07-24

## Overall assessment

**Recommendation: Reject in the current form (approximately 3/10).**

The central idea is potentially interesting, and the completed beam-width and multi-seed runs improve the empirical story. However, the submitted manuscript currently mixes results from different checkpoints, seeds, and evaluation protocols without labeling those differences. At least one main table and one appendix table still contain metrics from a known evaluation bug. Several figures are not publication-ready, and most figures are never explicitly referenced in the prose. These are credibility and presentation blockers rather than cosmetic issues.

## 1. Experiment authenticity and consistency

### Critical: stale metrics from a known evaluation bug

The project decision log records that the old CE-frozen result `R@10=0.2613` came from an evaluation protocol bug and that the corrected three-seed result is `0.3114` (seed 42: `0.3036`). The paper nevertheless reports:

- Table `tab:loss`: Softmax CE `0.2613`, described as a three-seed average.
- Table `tab:temperature`: CE values `0.2613`, `0.2611`, and `0.2373`, plus the obsolete claim of an approximately 20% validation-test gap.
- Figure 5: corrected values `0.3114`, `0.3070`, and `0.2846`.

The text, tables, and figure therefore cannot all be true under one protocol. Table `tab:loss`, Table `tab:temperature`, and the associated prose must be regenerated from the corrected evaluator before submission.

### Critical: same method names refer to different experimental objects

- `CE-frozen` is `0.3114` in the overall ML-1M table but `0.2613` in the loss table.
- `MSE-frozen` is `0.2482` in the overall table but `0.2425` in the cross-dataset and loss tables.
- `SASRec` is `0.2781` in the overall table but `0.2797` elsewhere.
- `TIGER` is `0.2992` in the main comparison but `0.3060` in the beam-width sweep at width 100.
- Reachability at width 100 is `63.8%` in the introduction/main analysis but `64.8%` in the sweep.

Some differences may be legitimate (different seed, checkpoint, or evaluation script), but the manuscript does not identify them. Every result row needs an experiment ID, seed count, checkpoint-selection rule, and evaluator version in the provenance record. A reviewer will otherwise interpret these as cherry-picking or irreproducibility.

### Major: mixed-seed and mixed-protocol comparisons

- The main ML-1M table compares three-seed CE-frozen against single-seed TIGER.
- The crossover plot uses CE seed 42 (`0.3036`) and TIGER `0.2992`, while the main table uses the CE three-seed mean (`0.3114`).
- The beam-width sweep appears to use a run distinct from the main TIGER result, but its seed/checkpoint is absent from the table and caption.
- The Sports multimodal comparison is single-seed and includes only about half of item images (`5,268/10,694` encoded according to the run report). The missing-image policy is not described in the paper.

These comparisons can remain, but captions must state their scope and the text must avoid implying uniform multi-seed evidence.

### Major: provenance is incomplete for a submission artifact

The GitHub paper repository contains the manuscript and figures but not a machine-readable mapping from each table/figure cell to result JSON, checkpoint, seed, and evaluation command. The external project provenance file also marks several claims as `pre-registry` rather than linking them to immutable artifacts. For a paper whose central contribution is empirical diagnosis, this is insufficient.

Minimum required provenance fields:

1. experiment ID and git commit;
2. dataset checksum and split procedure;
3. training seed and evaluation seed;
4. checkpoint selection criterion;
5. exact evaluator and exclusion protocol;
6. raw result JSON hash;
7. figure/table generation script.

## 2. Figures

### Figure 1

The LaTeX display width has been reduced from `1.0\textwidth` to `0.9\textwidth`.

Remaining quality issues in the source graphic:

- visible typo/placeholder-like text: `label: target item / SID...`;
- awkward title: `SID Based - GenRec` should be `SID-based generative recommendation`;
- inconsistent naming (`item_ids`, `embedding_1`, `SASRec Emb`);
- overlapping stacked target boxes obscure text;
- `High Quality Mapping` is vague and not defined;
- visual semantics are inconsistent: dotted arrows, dotted boxes, and colors lack a legend;
- the figure shows user dense/sparse features although the formal method is defined primarily from interaction histories.

This figure should be redrawn before submission; merely scaling it down will make its already dense labels harder to read.

### Figure 2

The crossover plot is generally legible, but it lacks explicit seed/checkpoint information. The legend states TIGER is `63.8% reachable`, while the separate beam sweep reports `64.8%` at width 100. The caption should identify the exact run and explain that the dashed quartile ceilings are raw union-of-beams coverage rather than Recall upper bounds under all possible user distributions.

### Figure 3

The figure is visually clean but largely duplicates Table `tab:reachability`. It is never explicitly cited in the prose. Either cite it and use it to replace the raw table, or remove it to save space. The phrase `CQG-Rec (all quartiles)` at 100% denotes theoretical scoreability, not empirically observed coverage; the axis/caption should make that distinction explicit.

### Figure 4

Not publication-ready:

- x-axis labels overlap severely;
- annotations overlap one another and the bars;
- `I-I=...` is undefined in the figure;
- variants `v3`, `v4`, `v5h`, `v6`, and `v7` are not meaningful without a mapping;
- the scratch baseline label intersects the reference line;
- near-zero bars are visually indistinguishable.

Replace it with a compact table or redesign it as two panels: representation collapse diagnostics and Recall@10.

### Figure 5

This is the strongest figure visually, but it exposes the stale-table problem because its corrected numbers disagree with Tables `tab:loss` and `tab:temperature`. The `0.03%` visibility labels also need a precise denominator and counting convention: SASRec/BPR sees one positive plus one negative, so the paper must specify whether visibility is `1/N`, `2/N`, or candidate-negative visibility.

## 3. Tables

- Several tables do not state seed count, standard-deviation convention, or evaluator protocol.
- The overall table mixes single-seed and three-seed rows; a footnote is preferable to prose hidden in the caption.
- The cross-dataset table labels `CQG standalone` without stating that the ML-1M value corresponds to MSE rather than the stronger CE variant.
- The loss table claims a three-seed average but contains the obsolete bugged CE result.
- The appendix beam table duplicates central evidence but omits latency even though the main text makes a latency claim.
- The automatically synchronized TIGER table says the main aggregation “should” use mean and standard deviation, signaling unfinished manuscript generation.
- The multimodal table uses internal run names (`sports-mm-cqg-ce-s42`) rather than reader-facing method names.

## 4. Cross-references and indexing

Automated checks found:

- no missing citation keys;
- no duplicate bibliography keys;
- no duplicate LaTeX labels;
- no references to nonexistent labels;
- all five figure files exist.

However, Figures 2–5 are not explicitly referenced by label in the prose. Several tables and appendix sections are also unreferenced. Valid labels are not enough: AAAI reviewers expect every figure/table to be introduced and interpreted in the text.

## 5. Citation authenticity

All 21 cited keys correspond to real works, but six bibliography entries contained incorrect metadata and were corrected during this audit:

- IDGenRec author list;
- DIGER second author;
- UniGRec author list;
- Expressiveness Limits author names;
- DreamRec title, full author list, and venue;
- ContRec author names.

One bibliography entry (`geng2022p5`) is uncited and should be removed unless used. The paragraph on “recent work” about popularity bias in SID systems currently makes a literature claim without a citation. Add the specific relevant source or narrow the statement.

Primary-source checks used official/arXiv/venue records for TIGER, DIGER, UniGRec, the expressiveness-limit paper, DreamRec, ContRec, OneRec, and IDGenRec.

## 6. Reproducibility and writing quality

Strengths:

- dataset splits and metrics are defined;
- the distinction between code uniqueness and inference reachability is clear;
- limitations acknowledge the k-means-versus-published-TIGER mismatch;
- the beam-width sweep directly addresses an obvious reviewer question.

Weaknesses:

- no public code or result-artifact link is provided in the manuscript;
- “TIGER-style” is easy to miss and may be read as a faithful TIGER reproduction;
- claims such as “massive, dynamic catalogs” exceed the demonstrated scale (largest main-table catalog is about 12K items);
- dynamic-catalog benefits are argued but not evaluated;
- ANN scalability is discussed without latency/memory measurements;
- the abstract does not report a quantitative headline result;
- the paper mixes a mechanism paper, a method paper, and a negative-results report, weakening the central narrative.

## Required actions before submission

1. Regenerate all CE tables from the corrected evaluator and delete obsolete bugged numbers.
2. Build one authoritative result manifest and generate every table/figure from it.
3. Label every result with seed count, checkpoint, beam width, and protocol.
4. Reconcile `63.8%` versus `64.8%`, `0.2992` versus `0.3060`, and the SASRec/MSE discrepancies.
5. Redesign Figures 1 and 4; explicitly cite every retained figure.
6. Replace internal run names and unfinished auto-generated wording.
7. Add code/data/result provenance or an anonymized supplementary archive.
8. Narrow scalability/dynamic-catalog claims unless supported by a larger-scale experiment.

