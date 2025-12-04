import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
from sklearn.model_selection import train_test_split
from model import Siamese1DNet, ContrastiveLoss
# ======================
# DATA AUGMENTATION
# ======================
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

def preprocess_for_1dcnn(img, target_length=200, augment=True):
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
    # Shape: (40, target_length, 3) -> (target_length, 3)
    avg_line = np.mean(norm, axis=0)
    
    # Transpose để có shape (3, target_length)
    line_1d = avg_line.T
    
    return np.expand_dims(line_1d, axis=0)  # (1, 3, target_length)

# ======================
# DATASET
# ======================
class ConnectorDataset(Dataset):
    def __init__(self, image_paths, labels, augment=True, aug_per_image=5):
        """
        image_paths: list các đường dẫn ảnh
        labels: list các nhãn (connector type)
        augment: có áp dụng augmentation không
        aug_per_image: số lần augment mỗi ảnh
        """
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.aug_per_image = aug_per_image
        
        # Load tất cả ảnh
        self.images = []
        self.valid_indices = []
        
        print("Loading images...")
        for idx, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is not None and img.size > 0:
                self.images.append(img)
                self.valid_indices.append(idx)
            else:
                print(f"⚠️ Cannot load: {img_path}")
        
        print(f"✅ Loaded {len(self.images)} valid images from {len(image_paths)} total")
        
    def __len__(self):
        if self.augment:
            return len(self.images) * self.aug_per_image
        return len(self.images)
    
    def __getitem__(self, idx):
        # Tìm ảnh gốc tương ứng
        real_idx = idx % len(self.images)
        img = self.images[real_idx]
        label = self.labels[self.valid_indices[real_idx]]
        
        # Preprocess với/không augment
        tensor = preprocess_for_1dcnn(img, augment=self.augment)
        tensor = torch.tensor(tensor).float().squeeze(0)  # (3, 250)
        
        return tensor, label

def create_pairs_from_batch(batch_data, batch_labels):
    """Tạo cặp positive và negative từ batch"""
    pairs_1 = []
    pairs_2 = []
    labels = []
    
    batch_size = len(batch_labels)
    
    # Tạo positive pairs (cùng class)
    unique_labels = torch.unique(batch_labels)
    for label in unique_labels:
        indices = (batch_labels == label).nonzero(as_tuple=True)[0]
        if len(indices) >= 2:
            # Chọn tất cả các cặp cùng class
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pairs_1.append(batch_data[indices[i]])
                    pairs_2.append(batch_data[indices[j]])
                    labels.append(0)  # 0 = same
    
    # Tạo negative pairs (khác class)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if batch_labels[i] != batch_labels[j]:
                pairs_1.append(batch_data[i])
                pairs_2.append(batch_data[j])
                labels.append(1)  # 1 = different
    
    if len(pairs_1) == 0:
        return None, None, None
    
    return torch.stack(pairs_1), torch.stack(pairs_2), torch.tensor(labels).float()

# ======================
# TRAINING
# ======================
def train_model(train_loader, val_loader, epochs=50, lr=0.001, save_path='siamese_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    model = Siamese1DNet().to(device)
    criterion = ContrastiveLoss(margin=1.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # ============ TRAINING ============
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Tạo pairs từ batch
            pairs_1, pairs_2, pair_labels = create_pairs_from_batch(batch_data, batch_labels)
            
            if pairs_1 is None:
                continue
            
            pairs_1 = pairs_1.to(device)
            pairs_2 = pairs_2.to(device)
            pair_labels = pair_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            distances = model(pairs_1, pairs_2)
            loss = criterion(distances, pair_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # ============ VALIDATION ============
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                pairs_1, pairs_2, pair_labels = create_pairs_from_batch(batch_data, batch_labels)
                
                if pairs_1 is None:
                    continue
                
                pairs_1 = pairs_1.to(device)
                pairs_2 = pairs_2.to(device)
                pair_labels = pair_labels.to(device)
                
                distances = model(pairs_1, pairs_2)
                loss = criterion(distances, pair_labels)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"✅ Model saved with val_loss: {avg_val_loss:.4f}")
    
    print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
    return model

# ======================
# MAIN
# ======================
def main():
    # ========== CẤU HÌNH ==========
    data_dir = "chamber"  # Thay đổi đường dẫn của bạn
    batch_size = 32
    epochs = 50
    learning_rate = 0.0008
    test_split = 0.1  # 10% cho test
    val_split = 0.15  # 15% cho validation
    
    # ========== LOAD DATASET ==========
    print("📂 Loading dataset...")
    image_paths = []
    labels = []
    class_names = []
    
    for label_idx, folder_name in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            class_names.append(folder_name)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(folder_path, img_name))
                    labels.append(label_idx)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total images: {len(image_paths)}")
    print(f"   Number of classes: {len(class_names)}")
    print(f"   Classes: {class_names}")
    
    if len(image_paths) == 0:
        print("❌ No images found! Please check your data directory.")
        return
    
    # ========== SPLIT TRAIN/VAL/TEST ==========
    # Bước 1: Tách test set trước
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, 
        test_size=test_split, 
        random_state=42, 
        stratify=labels
    )
    
    # Bước 2: Tách train và validation từ phần còn lại
    val_size_adjusted = val_split / (1 - test_split)  # Điều chỉnh tỷ lệ
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=train_val_labels
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training images: {len(train_paths)} ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"   Validation images: {len(val_paths)} ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"   Test images: {len(test_paths)} ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    # ========== LƯU TEST SET ==========
    print("\n💾 Saving test set information...")
    with open('test_set.txt', 'w', encoding='utf-8') as f:
        f.write("TEST SET\n")
        f.write("="*60 + "\n")
        for path, label in zip(test_paths, test_labels):
            f.write(f"{path}\t{class_names[label]}\n")
    print("✅ Test set saved to 'test_set.txt'")
    
    # ========== CREATE DATASETS ==========
    print("\n🔄 Creating datasets...")
    train_dataset = ConnectorDataset(
        train_paths, train_labels, 
        augment=True
    )
    val_dataset = ConnectorDataset(
        val_paths, val_labels, 
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # ========== TRAIN ==========
    print("\n🚀 Starting training...")
    model = train_model(train_loader, val_loader, epochs=epochs, lr=learning_rate)
    
    print("\n✅ Training completed!")
    print(f"✅ Model saved to 'siamese_model.pth'")
    print(f"✅ Test set saved to 'test_set.txt'")
    print(f"\n💡 Next steps:")
    print(f"   1. Run test: python test.py")
    print(f"   2. Use model: python use.py")

if __name__ == "__main__":
    main()