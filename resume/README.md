# Résumé (LaTeX)

Two variants of a two-page résumé, both built from the same custom LaTeX template
(serif body, small-caps ruled section headers, bold-title / right-aligned-date
entries). **Overleaf-ready** — upload a `.tex` and compile with **pdflatex**.

| Source | Output | Slant | Section order |
| --- | --- | --- | --- |
| `resume.tex` | `resume.pdf` | Research | Education, Experience, Publications, Projects, Honors, Talks, Skills |
| `infra_resume.tex` | `infra_resume.pdf` | Cloud infra / ML platform | Summary, Experience, Projects, Publications, Skills, Honors, Talks |

## Build

```bash
./build.sh    # pdflatex on resume.tex only
./deploy.sh   # builds every *.tex here, then commits + pushes to deploy
```

The site serves both PDFs directly. The sidebar shows a **CV** row with two pill
buttons wired up in `_config.yml`:

- `author.cv` → `/resume/resume.pdf`, labelled by `author.cv_label` ("Research")
- `author.cv_alt` → `/resume/infra_resume.pdf`, labelled by `author.cv_alt_label` ("Infra")

`author.cv_title` / `author.cv_alt_title` supply the hover tooltips. Dropping
`cv_alt` collapses the row back to a single plain **CV** link.

After editing a résumé, run `./deploy.sh` (or `./build.sh` plus a manual commit of
the PDF) to update the live links.

## Customize

Fonts, margins, spacing, and the section-header / entry macros are grouped in the
preamble of each `.tex`.
