import torch
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random

from model import Siamese1DNet

def augment_image(img):
    """Áp dụng random augmentation để tăng cường dữ liệu"""
    augmented = img.copy()
    
    # Random blur
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

    # Random shift (dịch chuyển nhỏ)
    if random.random() > 0.5:
        shift_x = random.randint(-5, 5)
        shift_y = random.randint(-5, 5)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented = cv2.warpAffine(augmented, M, (augmented.shape[1], augmented.shape[0]))
    
    return augmented

def preprocess_for_1dcnn(img, target_length=200, augment=False):
    """
    Preprocess ảnh thành input cho 1D CNN
    - Resize về kích thước chuẩn
    - Chuyển sang HSV
    - Normalize
    - Trung bình theo trục dọc để tạo 1D signal
    """
    if augment:
        img = augment_image(img)
    
    # Resize về kích thước chuẩn (width=30, height=target_length)
    resized = cv2.resize(img, (30, target_length))
    
    # Chuyển sang HSV color space
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Normalize về [0, 1]
    norm = hsv.astype(np.float32) / 255.0
    
    # Lấy trung bình theo trục dọc (axis=0) để tạo 1D signal
    avg_line = np.mean(norm, axis=0)
    
    # Transpose để có shape (3, target_length)
    line_1d = avg_line.T
    
    return np.expand_dims(line_1d, axis=0)  # (1, 3, target_length)

# ======================
# LOAD MODEL
# ======================
def load_model(model_path='siamese_model.pth'):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Siamese1DNet().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   Device: {device}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model, device

# ======================
# LOAD TEST SET
# ======================
def load_test_set(test_file='test_set.txt'):
    """Load test set từ file"""
    test_paths = []
    test_labels = []
    class_names = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]  # Bỏ header
        
        for line in lines:
            path, class_name = line.strip().split('\t')
            test_paths.append(path)
            
            if class_name not in class_names:
                class_names.append(class_name)
            
            test_labels.append(class_names.index(class_name))
    
    print(f"\n📂 Test Set Loaded:")
    print(f"   Total images: {len(test_paths)}")
    print(f"   Classes: {class_names}")
    
    return test_paths, test_labels, class_names

# ======================
# TEST MODEL
# ======================
def test_model(model, device, test_paths, test_labels, class_names, threshold=0.5):
    """
    Test model bằng cách so sánh mỗi ảnh với reference của class đó
    """
    print("\n🧪 Testing model...")
    print("="*60)
    
    # Tạo reference images (ảnh đầu tiên của mỗi class)
    references = {}
    for label in set(test_labels):
        class_name = class_names[label]
        # Tìm ảnh đầu tiên của class này
        for path, lbl in zip(test_paths, test_labels):
            if lbl == label:
                img = cv2.imread(path)
                if img is not None:
                    tensor = preprocess_for_1dcnn(img, augment=False)
                    references[label] = torch.tensor(tensor).float().to(device)
                    print(f"📌 Reference for '{class_name}': {path}")
                    break
    
    # Test từng ảnh
    predictions = []
    distances_list = []
    
    print("\n🔍 Processing test images...")
    for idx, (path, true_label) in enumerate(zip(test_paths, test_labels)):
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Cannot load: {path}")
            predictions.append(-1)
            distances_list.append([])
            continue
        
        # Preprocess
        test_tensor = torch.tensor(preprocess_for_1dcnn(img, augment=False)).float().to(device)
        
        # So sánh với tất cả references
        distances = {}
        with torch.no_grad():
            for label, ref_tensor in references.items():
                distance = model(test_tensor, ref_tensor).item()
                distances[label] = distance
        
        # Chọn class có distance nhỏ nhất
        predicted_label = min(distances, key=distances.get)
        min_distance = distances[predicted_label]
        
        predictions.append(predicted_label)
        distances_list.append(distances)
        
        # In kết quả
        status = "✅" if predicted_label == true_label else "❌"
        print(f"[{idx+1}/{len(test_paths)}] {status} True: {class_names[true_label]} | "
              f"Pred: {class_names[predicted_label]} | Dist: {min_distance:.4f}")
    
    # ========== METRICS ==========
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Confusion matrix saved to 'confusion_matrix.png'")
    
    # Distance Distribution
    plt.figure(figsize=(12, 6))
    
    correct_distances = []
    incorrect_distances = []
    
    for true_label, pred_label, dists in zip(test_labels, predictions, distances_list):
        if not dists:
            continue
        min_dist = min(dists.values())
        if true_label == pred_label:
            correct_distances.append(min_dist)
        else:
            incorrect_distances.append(min_dist)
    
    plt.hist(correct_distances, bins=30, alpha=0.6, label='Correct', color='green')
    plt.hist(incorrect_distances, bins=30, alpha=0.6, label='Incorrect', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distance_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Distance distribution saved to 'distance_distribution.png'")
    
    plt.show()
    
    return accuracy, predictions, distances_list

# ======================
# MAIN
# ======================
def main():
    print("="*60)
    print("MODEL TESTING")
    print("="*60)
    
    # Load model
    model, device = load_model('siamese_model.pth')
    
    # Load test set
    test_paths, test_labels, class_names = load_test_set('test_set.txt')
    
    # Test
    threshold = 0.5
    print(f"\n🎯 Using threshold: {threshold}")
    
    accuracy, predictions, distances = test_model(
        model, device, 
        test_paths, test_labels, class_names,
        threshold=threshold
    )
    
    print("\n" + "="*60)
    print(f"🎉 Testing completed!")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Results saved to:")
    print(f"      - confusion_matrix.png")
    print(f"      - distance_distribution.png")
    print("="*60)

if __name__ == "__main__":
    main()