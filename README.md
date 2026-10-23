# ROP Detection

Deep-learning screening for Retinopathy of Prematurity (ROP) in infant retinal photos. A ResNet-50 model classifies each photo as Normal or ROP; an eye visit is flagged as ROP if any of its photos is.

> Research prototype, not a medical device. All results must be reviewed by an ophthalmologist.

## Folders

| Folder | Contents |
|---|---|
| `notebooks/` | `rop-notebook.ipynb`: training and evaluation (run on Kaggle with a GPU) |
| `rop_app/` | Streamlit app: upload photos, get a prediction |
| `models/` | Trained model (`rop_detection_model.pth`, not in git) and its settings (`rop_model_config.json`) |
| `data/` | Kaggle dataset and clinic photos (not in git) |
| `docs/` | Older cloud-training guides |

## Run the app

```bash
pip install -r requirements.txt
streamlit run rop_app/app.py
```

The app needs `models/rop_detection_model.pth`. Download it (and `rop_model_config.json`) from the Kaggle notebook's output (`/kaggle/working`).

## Train the model

1. On Kaggle, import `notebooks/rop-notebook.ipynb`.
2. Attach the dataset `jananowakova/retinal-image-dataset-of-infants-and-rop` and the clinic photos dataset (upload `data/downloads/rop_test_images.zip`).
3. Turn on the GPU and click **Run All**.
4. Download `rop_detection_model.pth` and `rop_model_config.json` into `models/`.

## Data

- **Kaggle dataset:** Timkovič et al., *Retinal image dataset of infants and retinopathy of prematurity*, Scientific Data, 2024.
- **Clinic photos:** our own; half of the clinic babies are used for training, the other half for testing. A baby is never in both.
