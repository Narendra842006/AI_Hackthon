# Deployment

This repository contains a Streamlit app (`app.py`) and a `requirements.txt`.

Options to deploy:

- Streamlit Community Cloud (recommended):
  1. Go to https://streamlit.io/cloud and sign in with GitHub.
  2. Create a new app and select this repository and branch `main`.
  3. Set the main file to `app.py` and deploy.

- Container image (GitHub Container Registry):
  - A GitHub Actions workflow is included at `.github/workflows/docker-image.yml` that builds and pushes an image to `ghcr.io/<owner>/ai-triage-system:latest` on push to `main`.
  - To use the image, enable GitHub Packages (GHCR) and pull the image from your registry.

Note: Streamlit Cloud will deploy directly from this repository; no additional CI is required for Streamlit.
