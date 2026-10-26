import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import DEFAULT_IMAGE_SIZE, load_records, make_dataset

CLASSES_DICT = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution', 
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 
    41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

def compute_metrics(y_true, y_pred, num_classes=43):
    """
    Computes precision, recall, and f1 score using NumPy without requiring scikit-learn.
    """
    report = {}
    total_samples = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))

    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = int(np.sum(y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        report[CLASSES_DICT.get(c, str(c))] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    return accuracy, report

def plot_confusion_matrix(y_true, y_pred, num_classes=43, save_path="confusion_matrix.png"):
    """
    Computes and plots confusion matrix using pure NumPy and Matplotlib.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('GTSRB Traffic Sign - Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f"C{i}" for i in range(num_classes)], rotation=90, fontsize=8)
    plt.yticks(tick_marks, [f"C{i}" for i in range(num_classes)], fontsize=8)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved as: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate GTSRB Traffic Sign Model on Test Set")
    parser.add_argument("--model_path", type=str, default="traffic_sign_model.keras", help="Path to saved Keras model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where data is stored")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file '{args.model_path}' not found. Please run train.py first!")

    print(f"Loading trained model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    # Keep image decoding batched so the full test set does not consume excessive RAM.
    test_records = load_records(args.data_dir, "Test")
    test_dataset, y_test = make_dataset(
        test_records, DEFAULT_IMAGE_SIZE, args.batch_size
    )

    print("Running model predictions on the test set...")
    raw_preds = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(raw_preds, axis=1)

    acc, report = compute_metrics(y_test, y_pred, num_classes=43)

    print("\n=======================================================")
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print("=======================================================")

    # Save metrics report
    report_file = "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "class_report": report}, f, indent=4)
    print(f"Full evaluation report saved to {report_file}")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, num_classes=43)

if __name__ == "__main__":
    main()
