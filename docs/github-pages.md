# Publishing Docs on GitHub Pages

The documentation is built with MkDocs. Follow these steps to host it on GitHub Pages.

## 1. Install MkDocs and Theme

```bash
python3 -m pip install --user mkdocs mkdocs-material
```

## 2. Local Preview

```bash
mkdocs serve
```

Visit `http://127.0.0.1:8000/` to preview changes in real time.

## 3. Build Static Site

```bash
mkdocs build
```

The generated HTML appears in the `site/` directory.

## 4. Deploy Options

### Option A – `mkdocs gh-deploy`

MkDocs has a built-in deploy command that creates a gh-pages branch and pushes the site automatically.

```bash
mkdocs gh-deploy --force
```

This requires push access (or a personal access token if using CI).

### Option B – GitHub Actions

Create `.github/workflows/gh-pages.yml` with the following template:

```yaml
name: Deploy MkDocs

on:
  push:
    branches: [ main ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.10"

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install mkdocs mkdocs-material
      - name: Build site
        run: mkdocs build
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.DEPLOY_TOKEN }}
          publish_dir: site
```

**Important:** The default `GITHUB_TOKEN` doesn't have permission to push to the `gh-pages` branch. You need to create a Personal Access Token (PAT) with `repo` scope and add it as a repository secret named `DEPLOY_TOKEN`.

#### Creating a Personal Access Token

1. Go to [GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)](https://github.com/settings/tokens).
2. Click **Generate new token (classic)**.
3. Give it a descriptive name like "MkDocs Deploy".
4. Select the **repo** scope (full control of private repositories).
5. Click **Generate token**.
6. Copy the token immediately (you won't see it again).

#### Adding the Secret to Your Repository

1. Go to your repository's **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**.
3. Name: `DEPLOY_TOKEN`
4. Value: Paste your PAT.
5. Click **Add secret**.

Once merged, every push to `main` rebuilds and publishes the docs.

## 5. Enable GitHub Pages

1. Open the repository’s **Settings → Pages** section.
2. Select **Deploy from branch**, choose `gh-pages` and `/ (root)`.
3. Click **Save**.

The site will be live at `https://<username>.github.io/colab_dress` (or your custom domain).

## 6. Keep README in Sync

The root `README.md` links to the docs. Whenever you add or rename pages, update the navigation in `mkdocs.yml` so the site stays organized.
