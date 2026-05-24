# Scientific Poster

Conference poster draft for early July.

## Files

- `main.tex`: A0 portrait `beamerposter` source.
- `references.bib`: starter bibliography.
- `poster_notes.md`: content decisions, evidence, and remaining TODOs.
- `figures/`: local poster-only figures, if needed.

The poster currently references existing project plots directly from
`../results/cross_run_plots/` so the same generated evidence stays in one
place.

## Build

```bash
make
```

or:

```bash
latexmk -pdf main.tex
```

If `beamerposter` is missing, install it through the local TeX distribution
package manager before building.

## Before Submission

- Replace `TODO` conference, affiliation, and contact fields.
- Reconcile all numeric claims with final `results/`, `EVAL_MATRIX.md`, and
  MLflow.
- Export final plots at poster resolution if the embedded PNGs look soft.
- Add institution and conference logos only when usage rights are clear.
