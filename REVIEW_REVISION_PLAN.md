# Review Revision Plan

Source review: `/Users/sunyu/Desktop/GR_text/review.pdf`

## Completion Status

Completed for the final revision:

- ML-1M seeds 42--48 and Beauty/Sports seeds 42--44 were reselected by
  valid-prefix constrained validation and evaluated on the complete test sets.
- SID decoding now uses a catalog trie, exact code lookup, deterministic
  collision handling, history filtering, and adaptive internal
  over-generation. The obsolete numeric nearest-code fallback was removed
  from the shared implementation.
- All evaluated users receive the requested number of valid, distinct,
  history-filtered items; the evaluator records both returned-list length and
  internal beam size.
- Tables 2--4, Figures 2--3, the Abstract, Results, Discussion, Conclusion,
  and Appendices B/C/E were updated from the corrected results.
- Figure 1 now uses the paper's chronological-history terminology and depicts
  valid-prefix beam search. Figure 2 has no embedded title, and Figure 5's
  diagnostic label no longer overlaps its value.
- Sports is fully populated rather than marked pending. Its constrained TIGER
  R@10 is \(0.0236\pm0.0028\), and seed-42 CatalogCoverage@10/@100 is
  94.0\%/99.5\%.
- The final manuscript distinguishes user-level bootstrap uncertainty from
  training-run-level inference and uses TargetSupport only for SID
  beam-specific quantities.

This review is broadly positive, but its central numerical assessment is based
on the earlier TIGER decoding protocol. Revisions must therefore proceed in two
stages: first complete the constrained TIGER re-evaluation, then determine the
paper's main narrative from the corrected results. We must not strengthen the
claim that CQG-Single wins all seven seeds until the new TIGER results confirm
it.

## P0: Required Before Submission

### 1. Complete the constrained TIGER re-evaluation

Scope:

- ML-1M seeds 42–48.
- Beauty seeds 42–44.
- Sports seeds 42–44.
- For every seed, re-evaluate all saved `decoder_step*.pt` checkpoints on the
  constrained validation protocol.
- Select the best checkpoint by constrained validation R@10.
- Run the complete test evaluation using the newly selected checkpoint.

Recompute:

- R@10, NDCG@10, R@30, and R@90.
- TargetInOutput@100/200.
- TargetItemSupport.
- CatalogCoverage@10/100.
- Popularity-quartile metrics.
- Retrieval-depth curves.
- Effective returned-list length and internal beam size.
- Seven-seed mean, sample SD, paired difference, confidence interval, and
  exact sign-flip test.
- Seed-42 user-level paired bootstrap.

Affected paper components:

- Abstract.
- Main Results section.
- Tables 2–4.
- Figures 2–3.
- Appendices B, C, and E.
- Discussion and Conclusion.

Acceptance criteria:

- No nearest-code squared-distance mapping remains.
- A completed SID is accepted only through exact catalog lookup.
- Compared methods return the same number of valid, distinct,
  history-filtered items.
- Every formal result is traceable to a checkpoint selected by constrained
  validation.
- All regenerated figures and tables agree exactly with their source JSON
  results.

### 2. Reassess the main conclusion from the corrected TIGER results

Evaluate the following possible outcomes.

#### Outcome A: CQG-Single remains consistently stronger than TIGER

The paper may support a stronger positive conclusion:

> Autoregressive modeling does not require a discrete item vocabulary, and
> removing the discrete output interface need not reduce ranking accuracy.

#### Outcome B: CQG-Single and constrained TIGER are statistically comparable

Use a matching rather than dominance claim:

> Continuous query generation matches a properly constrained SID decoder while
> removing discrete code-generation and code-to-item infrastructure.

#### Outcome C: Constrained TIGER outperforms CQG-Single

Do not claim that removing the vocabulary improves accuracy. Use the controlled
factor-separation conclusion:

> Autoregression and discrete output vocabularies are separable design choices;
> continuous outputs trade some shallow accuracy for direct catalog
> scoreability and simpler retrieval infrastructure.

The CQG-AR factor-disentanglement contribution remains useful under all three
outcomes, but the paper must not assume in advance that accuracy improves after
removing the SID vocabulary.

### 3. Correct the TIGER decoding protocol throughout the paper

Method, Experimental Setup, and the appendix must consistently state:

- The catalog item codes are used to construct a valid-prefix SID trie.
- Every decoding step permits only valid next tokens under the current prefix.
- A completed SID must exactly match a catalog code.
- Code collisions use a deterministic handling rule.
- Previously observed items are removed.
- The evaluator over-generates or increases the internal beam after history
  filtering when necessary.
- Both nominal returned-list size and effective internal beam size are
  recorded.
- Checkpoints are selected using constrained validation.

Delete or replace:

- Squared numeric SID distance.
- Nearest-catalog-code fallback.
- The old no-refill implementation description.
- Any statement that describes an old unconstrained result as constrained
  decoding.

## P1: High-Value Text Revisions

### 4. Remove internal revision-history language

Replace:

> Figure 4 restores the broader diagnostic from the earlier manuscript.

With neutral forward-looking language:

> Figure 4 provides a broader diagnostic across the evaluated objectives.

Rewrite Appendix C.6 so that it explains positively why the paper uses
TargetInOutput, TargetItemSupport, and CatalogCoverage. Do not explain why an
earlier table was removed or discuss legacy target mappings.

Replace internal uses of:

- `legacy MSE checkpoints`
- `legacy CQG-Single MSE`
- `legacy MSE retriever`

With neutral descriptions:

- `MSE-trained checkpoints`
- `MSE-trained CQG-Single`
- `MSE-trained retriever`

Replace the concrete path `checkpoints/item_embeddings.pt` with:

> the shared frozen SASRec item-embedding table

### 5. Correct the significance statement in the Abstract

The Abstract must not report only the means while omitting the inferential
result. After the new experiment finishes, report:

- The seven-seed paired difference.
- The paired interval.
- The exact sign-flip p-value.
- An explicit statement that this is training-run-level inference.

Suggested structure:

> Across seven indexed ML-1M runs, ... with a mean paired difference of ..., a
> 95% paired interval of [...], and an exact sign-flip \(p=...\).

If the constrained TIGER result is no longer significant, remove any
significance implication from the Abstract.

### 6. Move the CQG-AR factor separation closer to the center of the narrative

The Abstract, Introduction contributions, and Conclusion should explain:

- CQG-AR is approximately parameter-matched to TIGER.
- CQG-AR retains serial autoregressive decoding.
- Its central change is replacing SID-token outputs with continuous outputs.
- CQG-Single separately tests whether serial refinement is necessary.

Recommended contribution order:

1. Factor separation between autoregressive structure and a discrete item
   vocabulary.
2. Finite-search diagnostics.
3. Controlled comparison of constrained SID and continuous retrieval.
4. Conditional crossover and retrieval-depth analysis.

The phrase “accuracy does not decrease and may improve after removing the
vocabulary” may be used only if the corrected TIGER results support it.

## P2: Presentation, Figures, and Readability

### 7. Fix explicit wording and layout issues

Replace:

> These steps introduce two practical bottlenecks: The first ...

With:

> These steps introduce two practical bottlenecks. The first ...

Confirm that `Conditional crossover pattern. In ...` contains the required
space after the period in the compiled PDF.

Format `Baselines` as a genuine heading, for example:

```latex
\paragraph{Baselines}
```

or:

```latex
\subsubsection{Baselines}
```

Do not allow it to appear as inline body text.

### 8. Verify Figure 1

The review's `item feuasteurre` report refers to an older version. The current
figure displays `item feature`, and `user impression history log` has been
changed to `chronological interaction history`.

The final PDF must still be checked for:

- Correct spelling of every `item feature` label.
- Legible font size.
- Agreement between the figure and the paper's terminology.
- An accurate depiction of valid-prefix trie constrained decoding.

### 9. Repair Figure 5

Regenerate Figure 5 from its source so that:

- `Scratch diagnostic baseline` does not overlap `0.248`.
- Labels and values have sufficient spacing.
- No text is corrupted or clipped.
- The figure remains vector-based rather than being repaired with a raster
  overlay.

### 10. Clarify nearby but differently weighted numbers

After the new data is available:

- Clearly distinguish the seed-42 beam-width-sensitivity value from the
  seven-seed mean and sample SD.
- Add the following clarification to the Figure 2 caption:

  > Values are user-weighted; Table 2 reports the corresponding item-balanced
  > quartile values.

- Retain the current item counts excluding the padding row:

  - ML-1M: 3,416.
  - Beauty: 12,101.
  - Sports: 10,693.

## Final Full-Paper Audit

### 1. Data audit

- Match every manuscript number to its result JSON or statistical-script
  output.
- Confirm that no result from the earlier decoding protocol remains in a
  formal table, figure, or claim.

### 2. Terminology audit

Maintain the distinctions among:

- TargetInOutput@K.
- TargetSupport@B.
- TargetItemSupport@K.
- CatalogCoverage@K.

Do not use TargetSupport as shorthand for a CQG TargetInOutput result.

### 3. Logical consistency audit

Ensure that the Abstract, Introduction, Results, Discussion, and Conclusion
make mutually consistent claims about:

- Method dominance or statistical comparability.
- The crossover pattern.
- User-level and training-run-level uncertainty.
- Checkpoint and search-budget dependence.

### 4. PDF visual audit

Render and inspect every page of the final PDF. Check:

- Table typography and numeric alignment.
- Figures 1, 2, 3, and 5.
- Captions.
- Float placement.
- Text overlap, corruption, clipping, and awkward line breaks.
- Citations and cross-references.

## Execution Order

1. [x] Finish the constrained TIGER evaluations.
2. [x] Determine the corrected main conclusion.
3. [x] Regenerate all affected statistics, Tables 2--4, and Figures 2--3.
4. [x] Correct the decoding protocol throughout the paper.
5. [x] Remove the internal revision-history language.
6. [x] Retain only CQG-AR claims supported by the corrected results.
7. [x] Repair the remaining typography and figure issues.
8. [x] Compile, render, and complete the final data, logic, terminology, and
   visual audits.
