# Traffic Sign Recognition Using Deep Learning

## Project Report

**Project:** Traffic Sign Recognition (GTSRB)  
**Model:** Convolutional Neural Network (CNN)  
**Framework:** TensorFlow / Keras  
**Application:** Streamlit web app  
**Repository:** [Programmer-pathak/Traffic_Sign_Detection](https://github.com/Programmer-pathak/Traffic_Sign_Detection)

## Abstract

This project builds an image classifier for 43 traffic-sign categories in the German Traffic Sign Recognition Benchmark (GTSRB). A CNN was trained from scratch using 64 × 64 RGB inputs, training-time data augmentation, class-weighted loss, and validation-based checkpoint selection. The selected model achieved **95.53% accuracy** and **94.97% macro F1** on the separate GTSRB test set used in this project.

A Streamlit interface loads the trained model and accepts an uploaded image, a camera capture, or one of the included sample signs. The application classifies an image that contains a prominent sign; it does not locate signs in a full road scene. Benchmark results do not establish safety or performance on signs from other countries or arbitrary real-world images.

## 1. Introduction

Traffic signs convey important road information, and correctly distinguishing visually similar signs is a computer-vision classification task. This project uses supervised learning to map a sign image to one of 43 GTSRB class labels. It includes a reproducible training and evaluation pipeline and a user interface for trying the resulting classifier.

### 1.1 Objectives

- Train a CNN from scratch on the GTSRB training data.
- Improve generalization with augmentation, regularization, class weighting, and early stopping.
- Keep the official test data separate from training and validation.
- Provide a simple image-upload, camera, and sample-based demo.
- Report measurable results and document the system's limits.

### 1.2 Scope

The classifier predicts one of the 43 GTSRB classes for an input image. It does **not** detect or crop a sign from a road scene, identify every country's sign system, or provide driving advice that should be relied upon for safety-critical decisions.

## 2. Dataset

The project uses the German Traffic Sign Recognition Benchmark (GTSRB), with 43 labeled classes. The local training CSV contained **39,209 images**. A deterministic, class-stratified split assigned **31,367 images to training** and **7,842 to validation**. The separate GTSRB test CSV supplied **12,630 images** for final evaluation.

The complete dataset is kept out of the GitHub repository because it is large. The tracked `data/samples/` images are small demo inputs, not the training or test set.

## 3. Methodology

### 3.1 Input pipeline

- Read image paths and labels from the dataset CSV files.
- Decode and resize images in batches to limit memory usage.
- Resize the full image to **64 × 64 pixels** and normalize RGB values to `[0, 1]`.
- Use the same image size and normalization for training, evaluation, and app inference.
- Shuffle the training dataset while preserving a deterministic split.
- Weight training examples by the inverse frequency of their class to reduce the impact of class imbalance.

The training pipeline resizes the full dataset image; it does not use the ROI coordinates to crop the sign. For app input, users should crop tightly around one sign because the model is a classifier, not an object detector.

### 3.2 Data augmentation

Augmentation is applied inside the model during training and is inactive during inference:

- Small random rotations.
- Random horizontal and vertical translations.
- Random zoom.
- Random contrast and brightness changes.
- No horizontal flip, to preserve the meaning of directional signs.

### 3.3 Model architecture

The model accepts 64 × 64 RGB images and returns probabilities for 43 classes:

1. Training-only augmentation layers.
2. Three convolutional blocks with **24, 48, and 96 filters**. Each block has two 3 × 3 convolutions with batch normalization and ReLU, followed by max pooling and dropout.
3. Global average pooling.
4. A 256-unit ReLU dense layer with L2 regularization and dropout.
5. A 43-unit softmax output layer.

L2 regularization, dropout, augmentation, and early stopping are used to reduce overfitting; they cannot guarantee that a model will never memorize or fail on new images.

### 3.4 Training configuration

| Setting | Value |
|---|---|
| Initialization | Trained from scratch |
| Maximum epochs in completed run | 20 |
| Batch size | 64 |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Loss | Sparse categorical cross-entropy |
| Random seed | 42 |
| Best validation checkpoint | Epoch 17 |
| Checkpoint selection | Highest validation accuracy |
| Learning-rate callback | Reduce by 0.5 after 3 validation-loss plateaus |
| Early stopping | Validation-loss patience of 8; restore best weights |

The training script's default maximum is 30 epochs. The completed model was trained with a 20-epoch limit and the best checkpoint was restored. Training ran on Windows with TensorFlow 2.21.0; TensorFlow did not detect a GPU in this environment, so training used the CPU.

## 4. Results

The final model was evaluated on the separate GTSRB test set, not on the training images.

| Metric | Result |
|---|---:|
| Test images | 12,630 |
| Overall accuracy | **95.53%** |
| Macro F1 | **94.97%** |
| Classes | 43 |

Macro F1 is the unweighted average of the per-class F1 scores, so each class contributes equally regardless of its number of test images.

The weakest class recalls in this evaluation included:

| Class | Recall |
|---|---:|
| Double curve | 63.33% |
| Speed limit (20 km/h) | 71.67% |
| Dangerous curve left | 75.00% |
| Bumpy road | 75.83% |

These class-level results show why overall accuracy alone is not enough to describe model performance.

### 4.1 Demo sample check

The app's ten included demo images were also checked using the same preprocessing path as app inference; all ten predicted labels matched their listed expected classes. These curated demo samples are a usability check, **not** an independent accuracy benchmark.

### 4.2 Evaluation artifacts

- `evaluation_report.json` — overall accuracy and per-class precision, recall, F1, and support.
- `confusion_matrix.png` — test-set confusion matrix.
- `training_history.png` — training and validation accuracy/loss curves.

## 5. Application

The Streamlit application provides:

- A demo mode with included sample signs.
- Image upload and camera capture.
- A 43-class sign catalog.
- A predicted class, confidence, and top-five probability display.

The app loads `traffic_sign_model.keras`. Uploaded images are converted to RGB, resized to the model's input dimensions, and normalized consistently with the model input pipeline. For best results, submit one centered, clearly visible sign that fills most of the image.

## 6. Run instructions

Use Python 3.13 and install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Run the web app from the project root:

```bash
streamlit run app.py
```

To retrain from scratch, first download and extract GTSRB under `data/` so that `Train.csv`, `Test.csv`, `Train/`, and `Test/` are available. Then run:

```bash
python src/train.py --epochs 30 --batch_size 64
```

Evaluate the saved model on the held-out test set with:

```bash
python src/evaluate.py
```

## 7. GitHub and deployment status

The source code and trained model are in the public GitHub repository. The complete dataset and local virtual environment are excluded. The repository contains instructions for a later Streamlit Community Cloud deployment. **The repository is published, but the Streamlit app itself is not deployed as a public hosted app.** GitHub Pages cannot run this Python/Streamlit application.

## 8. Limitations and future work

- The model only covers the 43 GTSRB classes and has no general-purpose or unknown-class output.
- It classifies a sign image; it cannot find signs within a full road scene.
- Resizing a large or differently shaped photo to a square can make a small sign hard to recognize. Cropping the sign helps, but does not guarantee a correct prediction.
- Test-set results on GTSRB do not guarantee performance on other countries, weather, cameras, distances, lighting, occlusion, or real driving conditions.
- Several classes have lower recall than the overall accuracy. More representative examples and targeted error analysis would help assess and improve those classes.
- Confidence scores are model probabilities, not guarantees of correctness or calibrated uncertainty.
- This project is an educational classifier, not a certified automotive system. Do not use its output to make driving decisions.

Potential next steps include evaluating on independently collected, geographically diverse images; adding a sign detector before classification; investigating low-recall classes; and calibrating confidence or adding an explicit abstain option.

## 9. Reproducibility and project files

- `src/preprocess.py` — dataset records, stratified split, and batched image pipeline.
- `src/model.py` — CNN architecture and training-only augmentation.
- `src/train.py` — training loop, class weights, callbacks, and history plot.
- `src/evaluate.py` — held-out test metrics and confusion matrix.
- `app.py` — Streamlit application.
- `traffic_sign_model.keras` — best saved model from the completed run.

