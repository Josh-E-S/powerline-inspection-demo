# Data directory

Everything in this directory except this file is gitignored.

## Getting InsPLAD

The Google Drive link in the InsPLAD GitHub README is dead (404 as of
2026-07-31). The working source is the authors' official deposit on
Mendeley Data:

- Landing page: https://data.mendeley.com/datasets/5n3fjgvfyz/1
- Direct download (single 6.4 GB zip containing all three components):
  https://data.mendeley.com/public-files/datasets/5n3fjgvfyz/files/96707044-99bb-40b2-bf23-6fa1b41ab9b0/file_downloaded

License: CC BY-NC 3.0 (attribution required, non-commercial). Fine for
non-commercial use with attribution; cite the paper (arXiv 2311.01619).

Steps:

1. Download the zip (resumable):

   ```
   curl -L -C - -o data/raw/InsPLAD_Dataset.zip \
     "https://data.mendeley.com/public-files/datasets/5n3fjgvfyz/files/96707044-99bb-40b2-bf23-6fa1b41ab9b0/file_downloaded"
   ```

2. Unzip into `data/raw/`. Expected contents (verify against the GitHub
   README): InsPLAD-det, supervised fault classification, and unsupervised
   anomaly detection. Only the first two are used by this project.

3. Run `python scripts/prep_insplad.py` from the repo root. It verifies the
   layout, converts detection annotations to YOLO txt format, and builds the
   train/val/test splits with a fixed seed.
