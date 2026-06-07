#!/usr/bin/env bash
# Compile the résumé. resume.pdf is served directly by the site at
# /resume/resume.pdf (the sidebar "CV" link), so commit it after building.
set -e
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:$PATH"
pdflatex -interaction=nonstopmode -halt-on-error resume.tex >/dev/null
echo "✓ Built resume.pdf — commit it to update the site's CV link"
