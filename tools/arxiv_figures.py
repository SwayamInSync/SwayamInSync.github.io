#!/usr/bin/env python3
"""
arxiv_figures.py — fetch a paper's arXiv source, pick its teaser figure, and
format a web-ready thumbnail (+ a ready-to-paste paper-box snippet) for the site.

Per paper it:
  1. Downloads the arXiv e-print source (or a PDF you point it at).
  2. Renders EVERY figure to PNG (Pillow for raster; pdftoppm/ImageMagick for
     PDF/EPS), saving them to tools/figure_candidates/<slug>/ so you can swap.
  3. Picks the most "teaser-like" one by image size + aspect ratio + filename
     keywords + how early it's referenced in the LaTeX — not just doc order, so
     mascots/logos/tiny icons lose to real architecture/results figures.
  4. Flattens onto white, resizes to a web width, writes images/<slug>.png.
  5. Falls back to rendering page 1 of the PDF if no usable figure is found.
  6. Prints the <div class='paper-box'> ... snippet for _pages/about.md.

Usage:
  python3 tools/arxiv_figures.py --batch tools/papers.json
  python3 tools/arxiv_figures.py 2305.06161 --slug starcoder --badge "TMLR 2023"
  python3 tools/arxiv_figures.py --pdf-url https://site/paper.pdf --slug nextcoder --badge "ICML 2025"

Requires: Pillow, and (for vector figures) pdftoppm/poppler + ImageMagick/Ghostscript.
"""
import argparse
import gzip
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

UA = {"User-Agent": "swayam-portfolio-figurebot/1.0 (personal academic site)"}
RASTER = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
VECTOR = {".pdf", ".eps", ".ps"}

TEASER_KW = re.compile(
    r"(teaser|pull|overview|framework|architect|pipeline|method|approach|"
    r"concept|system|flow|model|main|intro)",
    re.I,
)
PENALTY_KW = re.compile(
    r"(logo|icon|orcid|creativecommons|cc[\W_]?by|watermark|header|footer|"
    r"copyright|qr|email|badge|emoji|mascot|sticker|avatar)",
    re.I,
)


def log(msg):
    print(f"  {msg}", file=sys.stderr)


def download(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last


def get_metadata(arxiv_id):
    """Return (title, authors_list) from the arXiv API, or (None, [])."""
    try:
        data = download(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = ET.fromstring(data).find("a:entry", ns)
        if entry is None:
            return None, []
        title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
        authors = [
            " ".join(a.findtext("a:name", default="", namespaces=ns).split())
            for a in entry.findall("a:author", ns)
        ]
        return title, authors
    except Exception as e:  # noqa: BLE001
        log(f"metadata lookup failed: {e}")
        return None, []


def fetch_source(arxiv_id, workdir):
    """Download + extract the e-print source. Returns the extraction dir or None."""
    try:
        blob = download(f"https://arxiv.org/e-print/{arxiv_id}")
    except Exception as e:  # noqa: BLE001
        log(f"e-print download failed: {e}")
        return None
    if blob[:4] == b"%PDF":
        log("e-print is a PDF (no LaTeX source)")
        return None
    srcdir = workdir / "src"
    srcdir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            tf.extractall(srcdir)  # noqa: S202 (personal tool, trusted input)
        return srcdir
    except tarfile.ReadError:
        pass
    try:
        gzip.decompress(blob)
        log("source is a single gzipped file (no figure assets)")
    except Exception:  # noqa: BLE001
        log("could not parse e-print archive")
    return None


def find_main_tex(srcdir):
    best, best_score = None, -1
    for p in srcdir.rglob("*.tex"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        score = (
            (10 if "\\documentclass" in txt else 0)
            + (5 if "\\begin{document}" in txt else 0)
            + txt.count("\\includegraphics") * 0.1
            + len(txt) * 1e-6
        )
        if score > best_score:
            best, best_score = p, score
    return best


def reference_order(srcdir):
    """Map figure-stem(lower) -> first \\includegraphics index in the main tex."""
    main = find_main_tex(srcdir)
    if not main:
        return {}
    txt = main.read_text(errors="ignore")
    refs = re.findall(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}", txt)
    order = {}
    for i, r in enumerate(refs):
        stem = Path(r.strip().strip('"')).stem.lower()
        order.setdefault(stem, i)
    return order


def convert_to_png(src, out, width, density=200):
    """Render any supported figure to a white-background PNG of the given width."""
    ext = src.suffix.lower()
    try:
        if ext in RASTER:
            img = Image.open(src)
            img.load()
        elif ext == ".pdf":
            with tempfile.TemporaryDirectory() as td:
                pre = Path(td) / "fig"
                subprocess.run(
                    ["pdftoppm", "-png", "-r", str(density), "-f", "1", "-l", "1",
                     "-singlefile", str(src), str(pre)],
                    check=True, capture_output=True,
                )
                img = Image.open(f"{pre}.png")
                img.load()
        elif ext in {".eps", ".ps"}:
            with tempfile.TemporaryDirectory() as td:
                t = Path(td) / "fig.png"
                subprocess.run(
                    ["magick", "-density", str(density), str(src), "-flatten", str(t)],
                    check=True, capture_output=True,
                )
                img = Image.open(t)
                img.load()
        else:
            return False
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg.convert("RGB")
        if img.width > width:
            h = round(img.height * width / img.width)
            img = img.resize((width, h), Image.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, "PNG", optimize=True)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"convert failed for {src.name}: {e}")
        return False


def score_png(path, ref_order):
    """Higher = more teaser-like, judged on the *rendered* image."""
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001
        return -1e9
    s = 0.0
    if w < 220 or h < 120:           # icons / tiny logos
        s -= 60
    ar = w / h if h else 0
    s += min(w * h, 1_200_000) / 40000.0       # bigger figure ~ up to 30 pts
    if 1.2 <= ar <= 4.0:
        s += 25                                # landscape (architecture/teaser)
    elif ar < 0.8:
        s -= 15                                # tall/narrow
    elif ar > 5:
        s -= 20                                # banner strip
    stem = path.stem.lower()
    if TEASER_KW.search(stem):
        s += 40
    if PENALTY_KW.search(stem):
        s -= 150
    if stem in ref_order:
        s += max(0, 30 - ref_order[stem] * 8)  # referenced early
    return s


def format_authors(authors, me="swayam"):
    """Show only the first author and me (bolded): 'First, …, **Me**, et al.'."""
    if not authors:
        return "AUTHORS — fill in"
    idx = next((i for i, a in enumerate(authors) if me in a.lower()), None)
    if idx is None:
        return authors[0] + (", et al." if len(authors) > 1 else "")
    if idx == 0:                               # I'm (joint) first author
        out = f"**{authors[0]}**"
        if len(authors) > 1:
            out += ", " + authors[1]
        return out + (", et al." if len(authors) > 2 else "")
    sep = ", …, " if idx > 1 else ", "
    out = f"{authors[0]}{sep}**{authors[idx]}**"
    return out + (", et al." if idx < len(authors) - 1 else "")


def snippet(slug, badge, title, authors, arxiv_id):
    auth = format_authors(authors)
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    arxiv_link = f"[**arXiv**]({url})" if arxiv_id else ""
    return f"""<div class='paper-box'><div class='paper-box-image'><div><div class="badge">{badge}</div><img src='images/{slug}.png' alt="{slug}" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[{title or "TITLE — fill in"}]({url}) \\\\
{auth}

{arxiv_link}

- 🚧 add a one-line description
</div>
</div>
"""


def process(paper, out_dir, cand_root, width):
    slug = paper["slug"]
    badge = paper.get("badge", "")
    arxiv_id = paper.get("arxiv")
    pdf_url = paper.get("pdf_url")
    out = out_dir / f"{slug}.png"
    print(f"\n[{slug}] badge={badge!r} arxiv={arxiv_id} pdf_url={bool(pdf_url)}", file=sys.stderr)

    title, authors = (get_metadata(arxiv_id) if arxiv_id else (None, []))
    if authors:
        log(f"authors ({len(authors)}): " + ", ".join(authors))

    chosen = None
    with tempfile.TemporaryDirectory() as td:
        if arxiv_id:
            srcdir = fetch_source(arxiv_id, Path(td))
            if srcdir:
                ref_order = reference_order(srcdir)
                figs = [p for p in srcdir.rglob("*")
                        if p.is_file() and p.suffix.lower() in RASTER | VECTOR]
                cand_dir = cand_root / slug
                if cand_dir.exists():
                    shutil.rmtree(cand_dir)
                cand_dir.mkdir(parents=True, exist_ok=True)
                produced = []
                for f in figs:
                    name = f"{f.parent.name}__{f.stem}.png" if f.parent.name not in ("src", slug) else f"{f.stem}.png"
                    cp = cand_dir / name
                    if convert_to_png(f, cp, width):
                        produced.append((cp, f.stem.lower()))
                if produced:
                    chosen = max(produced, key=lambda t: score_png(t[0], ref_order))[0]
                    log(f"teaser pick: {chosen.name}  (from {len(produced)} candidates in {cand_dir})")

        if chosen:
            shutil.copyfile(chosen, out)
            log(f"wrote {out}")
        else:
            pdf_bytes = None
            try:
                if pdf_url:
                    pdf_bytes = download(pdf_url)
                elif arxiv_id:
                    pdf_bytes = download(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
            except Exception as e:  # noqa: BLE001
                log(f"pdf fetch failed: {e}")
            if pdf_bytes and pdf_bytes[:4] == b"%PDF":
                with tempfile.TemporaryDirectory() as td2:
                    p = Path(td2) / "p.pdf"
                    p.write_bytes(pdf_bytes)
                    pre = Path(td2) / "page"
                    try:
                        subprocess.run(
                            ["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1",
                             "-singlefile", str(p), str(pre)],
                            check=True, capture_output=True,
                        )
                        im = Image.open(f"{pre}.png").convert("RGB")
                        if im.width > width:
                            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        im.save(out, "PNG", optimize=True)
                        log(f"wrote {out}  (fallback: PDF first page — crop/replace if needed)")
                    except Exception as e:  # noqa: BLE001
                        log(f"first-page render failed: {e}")
                        return None
            else:
                log("FAILED — no figure produced; keep the placeholder + MARKER")
                return None

    print(snippet(slug, badge, title, authors, arxiv_id))
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch arXiv teaser figures for the site.")
    ap.add_argument("arxiv", nargs="?", help="arXiv id for single-paper mode")
    ap.add_argument("--slug")
    ap.add_argument("--badge", default="")
    ap.add_argument("--pdf-url")
    ap.add_argument("--batch", help="JSON file: [{slug, badge, arxiv?, pdf_url?}, ...]")
    ap.add_argument("--out", default="images")
    ap.add_argument("--candidates-dir", default="tools/figure_candidates")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_root = Path(args.candidates_dir)

    if args.batch:
        papers = json.loads(Path(args.batch).read_text())
    elif args.arxiv or args.pdf_url:
        if not args.slug:
            ap.error("--slug is required in single-paper mode")
        papers = [{"slug": args.slug, "badge": args.badge,
                   "arxiv": args.arxiv, "pdf_url": args.pdf_url}]
    else:
        ap.error("provide an arXiv id, --pdf-url, or --batch")

    if not shutil.which("pdftoppm"):
        log("WARNING: pdftoppm not found — PDF figures/fallbacks will fail.")

    print("\n===== paper-box snippets (paste into _pages/about.md) =====")
    for p in papers:
        process(p, out_dir, cand_root, args.width)


if __name__ == "__main__":
    main()
