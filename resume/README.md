# Résumé (LaTeX)

A single-page résumé built from a custom template that mirrors my original design
(centered name, FontAwesome contact row, ruled section headers, bold-title/right-date
entries). **Overleaf-ready** — upload `resume.tex` and compile with **pdflatex**.

## Build

```bash
./build.sh   # runs pdflatex, producing resume.pdf
```

The site serves this PDF directly — the sidebar **CV** link (`author.cv` in
`_config.yml`) points to **`/resume/resume.pdf`**. After editing the résumé, run
`./build.sh` and commit `resume.pdf` to update the live CV link.

## Customize

- **Accent color:** `\definecolor{accent}{HTML}{1A1A1A}` in the preamble.
- **Font size** (`9pt`, what keeps it to one page), **margins**, and **spacing** are
  grouped near the top of `resume.tex`.
