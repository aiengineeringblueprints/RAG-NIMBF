# Paper

LaTeX project for a paper about the benchmarking framework.

## Build

```bash
cd Paper
make
```

Direct build without `make`:

```bash
latexmk -pdf main.tex
```

Clean auxiliary files:

```bash
make clean
```

## Structure

- `main.tex`: paper entry point
- `sections/`: paper sections
- `references.bib`: bibliography
- `figures/`: plots and images for the paper
