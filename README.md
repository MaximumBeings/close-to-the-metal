# Close to the Metal — Website

Static site built with MkDocs Material.

## Open locally (no server needed)

Double-click `site/index.html` in Finder — or:

```bash
open site/index.html
```

## Serve with live reload (for editing)

```bash
pip install mkdocs mkdocs-material
mkdocs serve
# → http://127.0.0.1:8000
```

## Rebuild after editing chapters

```bash
mkdocs build
```

Edited source files live in `docs/`. The built HTML is in `site/`.

## Deploy to GitHub Pages (free hosting)

```bash
mkdocs gh-deploy
```

This pushes `site/` to the `gh-pages` branch automatically.
Your book will be live at `https://<your-username>.github.io/<repo-name>/`.
