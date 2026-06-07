# Résumé (LaTeX)

A single-page résumé built from a custom template that mirrors my original design
(centered name, FontAwesome contact row, ruled section headers, bold-title/right-date
entries). **Overleaf-ready** — upload `resume.tex` and compile with **pdflatex**.

## Build locally

```bash
./build.sh
```

This compiles `resume.tex` and copies the PDF to **`../files/Swayam-Singh-Resume.pdf`**,
which the website's sidebar **CV** link (`author.cv` in `_config.yml`) points to.

## Customize

- **Accent color:** `\definecolor{accent}{HTML}{1A1A1A}` in the preamble.
- **Font size** (`9pt`, what keeps it to one page), **margins**, and **spacing** are
  grouped near the top of `resume.tex`.

The `resume/` folder is excluded from the Jekyll build; only the compiled PDF in
`files/` is published.
