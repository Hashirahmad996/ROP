# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Retinopathy of Prematurity (ROP) detection: a PyTorch ResNet-50 classifier (Normal vs. ROP) for infant retinal photos, trained on Kaggle, plus a Streamlit app for predictions. The user prefers plain, simple explanations.

## Layout

```
data/                     # NOT in git (private patient photos, large)
  kaggle/                 # Kaggle dataset (Timkovic et al., Sci Data 2024), images_stack_without_captions/
  kaggle_sample/          # 100-image sample of the Kaggle dataset
  clinic/                 # Our clinic photos: plus pre, stage 3 neovascularization with ridge,
                          #   treated ROP (all ROP), normal vascularized retina (normal)
  downloads/              # Original zips; rop_test_images.zip = clinic folders, uploaded to Kaggle
models/
  rop_detection_model.pth # Current weights (download from Kaggle, NOT in git)
  rop_model_config.json   # Threshold + image size chosen during training (in git)
  old/                    # Outdated model (trained with wrong labels)
notebooks/
  rop-notebook.ipynb      # Training notebook (runs on Kaggle GPU)
rop_app/                  # Streamlit app (app.py UI, model_utils.py preprocessing + model)
docs/                     # Older cloud-training guides
```

## Key Facts

- **Labels** come from the `DG` code in Kaggle filenames (Table 10 of the dataset paper): 0 = normal; 2/3/4 = ROP stage 1/2/3; 8 = aggressive ROP; 9 = treated ROP. Skipped: 1 (stage 0) and 10-13 (other diseases, e.g. 11 = retinal hemorrhage, NOT normal).
- **Kaggle filenames**: `ID_sex_GA_BW_PA_DG_PF_D_S_imgNo.jpg` (baby ID = 1st part). The dataset stores each photo in 3 subfolders; the notebook de-duplicates by filename.
- **Clinic filenames**: `eye_frame_baby_date_time.jpg` (baby ID = 3rd part, prefixed `clinic_` in code). Half of the clinic babies are used for training, half for testing (seeded split).
- **Splits are by baby**, never by photo: train / validation (picks best epoch + threshold) / test.
- **Preprocessing** (must be identical in `notebooks/rop-notebook.ipynb` `ROPDataset.preprocess_image` and `rop_app/model_utils.py` `preprocess`): resize to 448, keep red+green channels, bilateral filter, CLAHE, ImageNet normalization.
- **Exam-level result**: an eye visit counts as ROP if any photo scores at or above the threshold (used in notebook evaluation and in the app).

## Commands

```bash
pip install -r requirements.txt

# Run the app (expects models/rop_detection_model.pth)
streamlit run rop_app/app.py
```

Training: import `notebooks/rop-notebook.ipynb` on Kaggle, attach the Kaggle dataset `jananowakova/retinal-image-dataset-of-infants-and-rop` and the clinic dataset (`data/downloads/rop_test_images.zip`), use a GPU, Run All. Download `rop_detection_model.pth` and `rop_model_config.json` from `/kaggle/working` into `models/`.
