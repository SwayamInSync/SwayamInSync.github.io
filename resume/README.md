# Résumé (LaTeX)

A clean, two-page résumé built from a custom LaTeX template (serif body, small-caps
ruled section headers, bold-title / right-aligned-date entries). Page 1 covers
Education, Experience, and Publications; page 2 covers Projects, Honors & Awards,
Invited Talks, and a Skills summary. **Overleaf-ready** — upload `resume.tex` and
compile with **pdflatex**.

## Build

```bash
./build.sh   # runs pdflatex, producing resume.pdf
```

The site serves this PDF directly — the sidebar **CV** link (`author.cv` in
`_config.yml`) points to **`/resume/resume.pdf`**. After editing the résumé, run
`./build.sh` and commit `resume.pdf` to update the live CV link.

## Customize

Fonts, margins, spacing, and the section-header / entry macros are grouped in the
preamble of `resume.tex`.
