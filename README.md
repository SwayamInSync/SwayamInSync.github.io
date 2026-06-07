# Swayam Singh — Personal Website

Source for my academic + industry homepage, served at <https://swayaminsync.github.io>.

Built on the [acad-homepage](https://github.com/RayeRen/acad-homepage.github.io) Jekyll theme
(itself based on [minimal-mistakes](https://github.com/mmistakes/minimal-mistakes) and
[academicpages](https://github.com/academicpages/academicpages.github.io)), MIT-licensed,
with a custom light/dark theme toggle and a Blog link to my writing.

## Editing content

Almost everything lives in three files:

- **`_pages/about.md`** — the entire homepage (About, Experience, Publications, Projects,
  Education, Awards). Just edit the Markdown.
- **`_config.yml`** — name, bio, location, avatar, and social links (GitHub, Google Scholar,
  LinkedIn, X, Kaggle, CV).
- **`_data/navigation.yml`** — the top navigation (section anchors + the external Blog link).

Profile photo: `assets/profile_picture.png`. Favicons: `favicon/`.
To show a **CV** link in the sidebar, drop a PDF in `files/` and set `author.cv` in `_config.yml`
(e.g. `cv: "/files/Swayam_Singh_CV.pdf"`).

## Light / dark mode

A custom toggle added on top of the theme:

- `_includes/head.html` sets the theme before CSS loads (prevents a flash of the wrong theme).
- `_includes/masthead.html` holds the toggle button.
- `_sass/_dark.scss` has the dark styles (imported last in `assets/css/main.scss`).
- `assets/js/dark-mode.js` handles the toggle and remembers your choice (`localStorage`),
  defaulting to your OS preference.

## Google Scholar citations (auto-updating)

`.github/workflows/google_scholar_crawler.yaml` crawls Google Scholar daily and pushes results
to a `google-scholar-stats` branch; the badge + per-paper counts read from there.

To enable it on your repo:

1. **Settings → Secrets and variables → Actions** → add `GOOGLE_SCHOLAR_ID` = `clLJfm8AAAAJ`.
2. Make sure `repository:` in `_config.yml` matches your repo.
3. **Actions** tab → run the **Get Citation Data** workflow once (it also runs daily).

## Run locally

Requires Ruby 3.x (3.3 recommended — the `github-pages` gem isn't compatible with Ruby 4 yet).

```bash
# with Homebrew's ruby@3.3:
export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
bundle install
bundle exec jekyll serve
# then open http://127.0.0.1:4000
```

## Deploy

This is a GitHub **user site**, so it deploys from the `SwayamInSync/SwayamInSync.github.io`
repository via GitHub Pages (build from the `main` branch). Push to `main` and Pages rebuilds
automatically — no extra build step needed.

## Credits

Theme: [acad-homepage](https://github.com/RayeRen/acad-homepage.github.io) ·
[minimal-mistakes](https://github.com/mmistakes/minimal-mistakes). MIT License (see `LICENSE`).
