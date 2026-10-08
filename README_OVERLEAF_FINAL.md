# Final AAAI 2027 Overleaf Package

Main file:

```text
AnonymousSubmission2027.tex
```

Appendix file:

```text
AnonymousSubmission2027_appendix.tex
```

Both files are independent compilation roots:

- `AnonymousSubmission2027.tex` produces only the main paper and References.
- `AnonymousSubmission2027_appendix.tex` produces only the Appendix.
- Neither root inputs the other file or depends on the other root's `.aux` file.

Compile setting:

```text
Compiler: pdfLaTeX
Bibliography: BibTeX
```

This project keeps the AAAI 2027 anonymous submission format:

- `\usepackage[submission]{aaai2027}`
- author set to `Anonymous Submission`
- affiliations left empty

The final paper title is:

```text
Beyond Semantic IDs: Finite-Beam Output Support in Generative Recommendation
```

Protocol-critical reproduction scripts:

```text
scripts/tiger_standalone.py
scripts/tiger_constrained_eval.py
scripts/tiger_select_constrained_checkpoint.py
```

The SID evaluator constructs a catalog trie, masks invalid prefixes at every
decoding step, uses exact code lookup with deterministic collision ordering,
filters history items, and expands the internal beam until the requested
returned-list size is filled or the documented safety cap is reached. Numeric
nearest-code fallback is not used.

The main draft is based primarily on `CQG-Rec_draft_Jul18.pdf`, with cross-dataset results, LLM backbone analysis, loss ablations, and appendix details integrated from `CQG-Rec-main+appendix.pdf`.
