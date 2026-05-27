# PSRL Documentation Guide

This document explains how to build, preview, and deploy the PSRL documentation, and describes what each file in the `docs/` directory does.

---

## Quick Start (Local Build)

```bash
# Install documentation dependencies
python -m pip install -r docs/requirements.txt

# Build HTML documentation
cd docs
make html

# Preview in browser
python -m http.server -d _build/html 8000
# Visit http://localhost:8000
```

---

## Deploying to ReadTheDocs

### Step 1: Push to GitHub

Ensure your repository is hosted on GitHub (or GitLab/Bitbucket) with the `docs/` directory and `.readthedocs.yaml` at the project root.

### Step 2: Import Project on ReadTheDocs

1. Go to [readthedocs.org](https://readthedocs.org/) and sign in with GitHub
2. Click **"Import a Project"**
3. Select your PSRL repository
4. ReadTheDocs will auto-detect `.readthedocs.yaml` and use it for build configuration

### Step 3: Configure Build Settings

- **Default branch**: `main` (or your primary branch)
- **Build on PR**: Enable "Build pull requests" for preview builds
- **Versions**: Optionally enable tagged versions for release docs

### Step 4: Verify

After the first build completes (usually 2-3 minutes):
- Visit `https://psrl.readthedocs.io/en/latest/`
- Check that navigation works, figures render, and code blocks have copy buttons

### Optional: Custom Domain

1. Add a CNAME record: `docs.your-domain.com` → `readthedocs.io`
2. In RTD project settings → Domains → Add `docs.your-domain.com`
3. RTD will provision an SSL certificate automatically

---

## How To...

### Add a New Page

1. Create a new `.md` file in the appropriate directory (e.g., `docs/design/new_feature.md`)
2. Add it to the parent's toctree. For example, in `docs/design/index.md`:
   ```markdown
   ```{toctree}
   :maxdepth: 1

   architecture
   parameter_server
   new_feature        ← add here
   ```
   ```
3. Build locally to verify: `cd docs && make html`

### Update Figures

1. Convert the paper PDF to SVG:
   ```bash
   # Option A: pdf2svg (Linux)
   pdf2svg input.pdf output.svg

   # Option B: Inkscape CLI
   inkscape input.pdf --export-filename=output.svg

   # Option C: Adobe Illustrator / Affinity Designer (manual export)
   ```
2. Place the SVG in `docs/_static/img/` with the expected filename
3. The documentation will automatically pick it up on next build

### Add Mermaid Diagrams

In any `.md` file, use:
````markdown
```{mermaid}
graph LR
    A[Train Worker] -->|push| B[Parameter Server]
    B -->|pull| C[Rollout Instance]
```
````

### Use Sphinx Design Components

Cards:
```markdown
:::{grid-item-card} Title
:link: target/page
:link-type: doc
Description text.
:::
```

Tabs:
```markdown
::::{tab-set}
:::{tab-item} Tab 1
Content for tab 1.
:::
:::{tab-item} Tab 2
Content for tab 2.
:::
::::
```

Admonitions:
```markdown
:::{admonition} Title
:class: tip
Content here.
:::
```

---

## CI Integration (Optional)

Add this GitHub Actions workflow for build-on-PR validation:

```yaml
# .github/workflows/docs.yml
name: Documentation Build Check
on: [pull_request]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r docs/requirements.txt
      - run: cd docs && make html
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `myst_parser` import error | `pip install myst-parser>=3.0` |
| Theme not found | `pip install pydata-sphinx-theme>=0.15` |
| Mermaid not rendering | Ensure `sphinxcontrib-mermaid` is in requirements.txt and conf.py extensions |
| RTD build fails | Check `.readthedocs.yaml` syntax, ensure requirements.txt has all deps |
| Broken cross-references | Use `{doc}` for page links: `` {doc}`../design/architecture` `` |
| Figures not showing | Verify SVG files exist in `docs/_static/img/` and paths in markdown start with `/_static/img/` |
| Local build warnings | Run `make html` and fix any `toctree` or reference warnings |
