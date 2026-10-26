# Traffic Sign Recognition (GTSRB)

A TensorFlow CNN that classifies an image containing one prominent traffic sign into one of the 43 German Traffic Sign Recognition Benchmark (GTSRB) classes. It is a classifier, not an object detector, and is not a driving-safety system.

## Setup

Python 3.13 is recommended. Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

The full dataset is intentionally not included in this repository. Download the GTSRB training/test archive separately and place its extracted files under `data/`:

```text
data/
  Train/                 # class folders 0–42
  Test/                  # test images
  Train.csv
  Test.csv
```

The small `data/samples/` images used by the app's demo mode are included. The trained model is also included, so the app can run without downloading the full training dataset.

## Train from scratch

Run from the project root:

```bash
python src/train.py --epochs 30 --batch_size 64
```

The pipeline decodes images in batches to limit RAM use, resizes the full image to 64×64, and creates a deterministic stratified train/validation split. The CNN uses training-only rotation, translation, zoom, brightness and contrast augmentation (no horizontal flip), batch normalization, dropout, L2 regularization, AdamW, class weighting, learning-rate reduction, and early stopping. The best validation-accuracy model is saved in native Keras format as `traffic_sign_model.keras`; curves are saved as `training_history.png`.

The validation split is used for model selection. Keep the separate GTSRB test set for the final evaluation:

```bash
python src/evaluate.py
```

Evaluation prints overall accuracy and per-class precision/recall/F1, and writes `evaluation_report.json` and `confusion_matrix.png`.

## Run the app

```bash
streamlit run app.py
```

## GitHub

The app is a Streamlit application; GitHub Pages cannot host its Python runtime. This repository contains the app source and trained model, but does not include the large dataset, virtual environment, or local secrets. After pushing the repository to GitHub, the app can later be deployed through Streamlit Community Cloud by selecting `app.py` as the entry point.

## Capstone report

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the detailed capstone report in Markdown, or open [PROJECT_REPORT.docx](PROJECT_REPORT.docx) for the Word version ready to edit/print. The report includes the system architecture, implementation, execution outputs, benchmark results, limitations, and references.

For uploads or camera captures, place one sign near the center and make it fill most of the image. The classifier does not find signs within a full road scene. Its classes are limited to GTSRB, so it cannot identify arbitrary signs from every country, and benchmark performance does not guarantee real-world accuracy. Do not rely on predictions for driving decisions.
