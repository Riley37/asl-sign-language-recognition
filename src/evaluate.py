import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image


def load_test_images(test_dir, img_size=(200, 200)):
    test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_files.sort()
    
    valid_letters = set(chr(i) for i in range(ord('A'), ord('Z') + 1))
    images, labels, valid_files = [], [], []
    
    for filename in test_files:
        class_name = filename.split('_')[0]
        if class_name in valid_letters:
            img_path = os.path.join(test_dir, filename)
            img = Image.open(img_path).convert('RGB').resize(img_size)
            images.append(np.array(img))
            labels.append(class_name)
            valid_files.append(filename)
    
    return np.array(images), labels, valid_files

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('ASL Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/asl_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(images, true_labels, pred_labels, pred_probs, filenames):
    num_images = len(images)
    cols = 6
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(18, 3*rows))
    
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f'{filenames[i]}\nTrue: {true_labels[i]}\nPred: {pred_labels[i]}\nConf: {pred_probs[i]:.3f}', 
                 color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/asl_test_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    TEST_DIR = 'data/test'
    MODEL_PATH = 'models/best_asl_model.h5'
    IMG_SIZE = (200, 200)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
    
    # Load class names
    class_names_file = 'models/asl_class_names.txt'
    if os.path.exists(class_names_file):
        with open(class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    model = keras.models.load_model(MODEL_PATH)
    test_images, true_labels, filenames = load_test_images(TEST_DIR, IMG_SIZE)
    
    # Make predictions
    predictions = model.predict(test_images, verbose=1)
    pred_indices = np.argmax(predictions, axis=1)
    pred_probs = np.max(predictions, axis=1)
    pred_labels = [class_names[idx] for idx in pred_indices]
    
    # Calculate accuracy
    true_indices = [class_names.index(label) for label in true_labels if label in class_names]
    correct = sum(1 for true_idx, pred_idx in zip(true_indices, pred_indices) if true_idx == pred_idx)
    accuracy = correct / len(true_indices)
    
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {correct}/{len(true_indices)}")
    
    # Show individual results
    for filename, true_label, pred_label, confidence in zip(filenames, true_labels, pred_labels, pred_probs):
        status = "✓" if true_label == pred_label else "✗"
        print(f"{status} {filename:15} | True: {true_label} | Pred: {pred_label} | Conf: {confidence:.3f}")
    
    # Generate reports and plots
    print(classification_report(true_indices, pred_indices, target_names=class_names, zero_division=0))
    plot_confusion_matrix(true_indices, pred_indices, class_names)
    visualize_predictions(test_images, true_labels, pred_labels, pred_probs, filenames)

if __name__ == "__main__":
    main()