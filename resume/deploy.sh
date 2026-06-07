#!/usr/bin/env bash
# Build the résumé PDF from the .tex source, then commit and push to deploy.
# GitHub Pages builds the site from `main`, so pushing updates the live CV
# link (/resume/resume.pdf). Run from anywhere: ./resume/deploy.sh
set -euo pipefail

RESUME_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$RESUME_DIR/.." && pwd)"
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"

# --- build every .tex in this folder (runs twice for stable refs/page count) ---
cd "$RESUME_DIR"
shopt -s nullglob
for tex in *.tex; do
  echo "→ Building ${tex%.tex}.pdf"
  pdflatex -interaction=nonstopmode -halt-on-error "$tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$tex" >/dev/null
done
# strip LaTeX build artifacts, keep only sources + PDFs
rm -f ./*.aux ./*.out ./*.log ./*.toc ./*.fls ./*.fdb_latexmk
echo "✓ Built: $(ls -1 *.pdf | tr '\n' ' ')"

# --- commit + push to deploy ---
cd "$REPO_DIR"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git add resume/

if git diff --cached --quiet; then
  echo "• No résumé changes to commit."
else
  git commit -m "Résumé: rebuild PDF and deploy"
  echo "✓ Committed résumé changes."
fi

echo "→ Pushing $BRANCH to origin"
git push origin "$BRANCH"
echo "✓ Deployed. GitHub Pages will rebuild the site shortly."
