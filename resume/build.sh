#!/usr/bin/env bash
# Build the résumé and copy the PDF into the site's files/ folder,
# where the website's sidebar "CV" link points.
set -e
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:$PATH"
pdflatex -interaction=nonstopmode -halt-on-error resume.tex >/dev/null
cp resume.pdf ../files/Swayam-Singh-Resume.pdf
echo "✓ Built resume.pdf → files/Swayam-Singh-Resume.pdf"
