import torch
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from PIL import Image
import os

from model_2d import SiameseMobileNet

# ======================
# TRANSFORMS
# ======================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================
# LOAD MODEL
# ======================
def load_model(model_path='siamese_mobilenet_model.pth'):
    """Load trained 2D model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseMobileNet(embedding_dim=128, pretrained=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   Device: {device}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model, device

# ======================
# PREPROCESS IMAGE
# ======================
def preprocess_image(img_path, transform):
    """Preprocess image for 2D CNN"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img)
    
    return img.unsqueeze(0)  # Add batch dimension

# ======================
# LOAD TEST SET
# ======================
def load_test_set(test_file='test_set_2d.txt'):
    """Load test set từ file"""
    test_paths = []
    test_labels = []
    class_names = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]  # Bỏ header
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
                
            path, class_name = parts
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
    print("\n🧪 Testing 2D MobileNet model...")
    print("="*60)
    
    # Tạo reference images (ảnh đầu tiên của mỗi class)
    references = {}
    reference_paths = {}
    
    for label in set(test_labels):
        class_name = class_names[label]
        # Tìm ảnh đầu tiên của class này
        for path, lbl in zip(test_paths, test_labels):
            if lbl == label:
                img_tensor = preprocess_image(path, test_transform)
                if img_tensor is not None:
                    references[label] = img_tensor.to(device)
                    reference_paths[label] = path
                    print(f"📌 Reference for '{class_name}': {os.path.basename(path)}")
                    break
    
    # Test từng ảnh
    predictions = []
    distances_list = []
    
    print("\n🔍 Processing test images...")
    for idx, (path, true_label) in enumerate(zip(test_paths, test_labels)):
        img_tensor = preprocess_image(path, test_transform)
        if img_tensor is None:
            print(f"⚠️ Cannot load: {path}")
            predictions.append(-1)
            distances_list.append([])
            continue
        
        img_tensor = img_tensor.to(device)
        
        # So sánh với tất cả references
        distances = {}
        with torch.no_grad():
            for label, ref_tensor in references.items():
                distance = model(img_tensor, ref_tensor).item()
                distances[label] = distance
        
        # Chọn class có distance nhỏ nhất
        predicted_label = min(distances, key=distances.get)
        min_distance = distances[predicted_label]
        
        predictions.append(predicted_label)
        distances_list.append(distances)
        
        # In kết quả
        status = "✅" if predicted_label == true_label else "❌"
        print(f"[{idx+1}/{len(test_paths)}] {status} True: {class_names[true_label]:15} | "
              f"Pred: {class_names[predicted_label]:15} | Dist: {min_distance:.4f}")
    
    # ========== METRICS ==========
    print("\n" + "="*60)
    print("📊 TEST RESULTS (2D MobileNet Model)")
    print("="*60)
    
    # Lọc predictions hợp lệ
    valid_indices = [i for i, p in enumerate(predictions) if p != -1]
    valid_test_labels = [test_labels[i] for i in valid_indices]
    valid_predictions = [predictions[i] for i in valid_indices]
    
    if len(valid_predictions) == 0:
        print("❌ No valid predictions!")
        return 0.0, [], []
    
    accuracy = accuracy_score(valid_test_labels, valid_predictions)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Total tested: {len(valid_predictions)}/{len(test_paths)}")
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(valid_test_labels, valid_predictions, 
                                target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(valid_test_labels, valid_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (2D MobileNet Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_2d.png', dpi=300, bbox_inches='tight')
    print("\n✅ Confusion matrix saved to 'confusion_matrix_2d.png'")
    
    # Distance Distribution
    plt.figure(figsize=(12, 6))
    
    correct_distances = []
    incorrect_distances = []
    
    for true_label, pred_label, dists in zip(valid_test_labels, valid_predictions, 
                                              [distances_list[i] for i in valid_indices]):
        if not dists:
            continue
        min_dist = min(dists.values())
        if true_label == pred_label:
            correct_distances.append(min_dist)
        else:
            incorrect_distances.append(min_dist)
    
    if correct_distances:
        plt.hist(correct_distances, bins=30, alpha=0.6, label='Correct', color='green')
    if incorrect_distances:
        plt.hist(incorrect_distances, bins=30, alpha=0.6, label='Incorrect', color='red')
    
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution (2D MobileNet Model)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distance_distribution_2d.png', dpi=300, bbox_inches='tight')
    print("✅ Distance distribution saved to 'distance_distribution_2d.png'")
    
    # Per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    for label, class_name in enumerate(class_names):
        class_indices = [i for i, l in enumerate(valid_test_labels) if l == label]
        if len(class_indices) == 0:
            continue
        
        class_predictions = [valid_predictions[i] for i in class_indices]
        class_accuracy = sum([1 for p in class_predictions if p == label]) / len(class_predictions)
        print(f"   {class_name:15}: {class_accuracy*100:5.1f}% ({len(class_indices)} images)")
    
    plt.show()
    
    return accuracy, valid_predictions, distances_list

# ======================
# COMPARE WITH 1D MODEL
# ======================
def compare_with_1d_results():
    """So sánh kết quả với 1D model nếu có"""
    if not os.path.exists('test_results_1d.txt'):
        return
    
    print("\n" + "="*60)
    print("📊 COMPARISON WITH 1D MODEL")
    print("="*60)
    
    # Đọc kết quả 1D model
    with open('test_results_1d.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())

# ======================
# MAIN
# ======================
def main():
    print("="*60)
    print("2D MOBILENET MODEL TESTING")
    print("="*60)
    
    # Load model
    model, device = load_model('siamese_mobilenet_model.pth')
    
    # Load test set
    test_paths, test_labels, class_names = load_test_set('test_set_2d.txt')
    
    # Test
    threshold = 0.5
    print(f"\n🎯 Using threshold: {threshold}")
    
    accuracy, predictions, distances = test_model(
        model, device, 
        test_paths, test_labels, class_names,
        threshold=threshold
    )
    
    # Lưu kết quả
    with open('test_results_2d.txt', 'w', encoding='utf-8') as f:
        f.write("2D MOBILENET MODEL TEST RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Total images: {len(test_paths)}\n")
        f.write(f"Model: MobileNetV2 (pretrained)\n")
        f.write(f"Embedding dim: 128\n")
    
    print("\n" + "="*60)
    print(f"🎉 Testing completed!")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Results saved to:")
    print(f"      - test_results_2d.txt")
    print(f"      - confusion_matrix_2d.png")
    print(f"      - distance_distribution_2d.png")
    print("="*60)
    
    # So sánh với 1D model nếu có
    compare_with_1d_results()

if __name__ == "__main__":
    main()