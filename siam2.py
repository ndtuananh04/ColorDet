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

# ======================
# PHÁT HIỆN ĐẦU NỐI
# ======================
def detect_metal_connectors(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metal_pins = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 2000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.3 < aspect_ratio < 3.0:
                metal_pins.append((x, y, w, h))

    if len(metal_pins) == 0:
        return None

    all_x = [x for x, y, w, h in metal_pins]
    all_y = [y for x, y, w, h in metal_pins]
    all_x_max = [x + w for x, y, w, h in metal_pins]
    all_y_max = [y + h for x, y, w, h in metal_pins]

    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max(all_x_max)
    max_y = max(all_y_max)

    extension = 160
    min_x = max(0, min_x - extension)
    max_x = min_x + 30
    padding = 20
    min_y = max(0, min_y - padding)
    max_y = min(frame.shape[0], max_y + padding)

    width = max_x - min_x
    height = max_y - min_y

    return (min_x, min_y, width, height), metal_pins

def crop_main_region(image):
    result = detect_metal_connectors(image)
    if result is None:
        return None
    (x, y, w, h), _ = result
    cropped = image[y:y+h, x:x+w]
    return cropped

# ======================
# DATA AUGMENTATION
# ======================
def augment_image(img):
    """
    Áp dụng random augmentation để tăng cường dữ liệu
    """
    augmented = img.copy()
    
    # Random blur
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)
    
    # Random shift (dịch chuyển nhỏ)
    if random.random() > 0.5:
        shift_x = random.randint(-3, 3)
        shift_y = random.randint(-3, 3)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented = cv2.warpAffine(augmented, M, (augmented.shape[1], augmented.shape[0]))
    
    # Random rotation (góc nhỏ)
    if random.random() > 0.5:
        angle = random.uniform(-5, 5)
        h, w = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h))
    
    return augmented

def preprocess_for_1dcnn(cropped, target_length=250, augment=False):
    if augment:
        cropped = augment_image(cropped)
    
    resized = cv2.resize(cropped, (target_length, 40))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    norm = hsv.astype(np.float32) / 255.0
    avg_line = np.mean(norm, axis=0)
    line_1d = avg_line.T
    return np.expand_dims(line_1d, axis=0)

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
        
        # Load và crop tất cả ảnh trước
        self.cropped_images = []
        self.valid_indices = []
        
        for idx, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is not None:
                cropped = crop_main_region(img)
                if cropped is not None:
                    self.cropped_images.append(cropped)
                    self.valid_indices.append(idx)
        
        print(f"Loaded {len(self.cropped_images)} valid images from {len(image_paths)} total")
        
    def __len__(self):
        if self.augment:
            return len(self.cropped_images) * self.aug_per_image
        return len(self.cropped_images)
    
    def __getitem__(self, idx):
        # Tìm ảnh gốc tương ứng
        real_idx = idx % len(self.cropped_images)
        cropped = self.cropped_images[real_idx]
        label = self.labels[self.valid_indices[real_idx]]
        
        # Preprocess với/không augment
        tensor = preprocess_for_1dcnn(cropped, augment=self.augment)
        tensor = torch.tensor(tensor).float().squeeze(0)  # (3, 250)
        
        return tensor, label

def create_pairs_from_batch(batch_data, batch_labels):
    """
    Tạo cặp positive và negative từ batch
    """
    pairs_1 = []
    pairs_2 = []
    labels = []
    
    batch_size = len(batch_labels)
    
    # Tạo positive pairs
    unique_labels = torch.unique(batch_labels)
    for label in unique_labels:
        indices = (batch_labels == label).nonzero(as_tuple=True)[0]
        if len(indices) >= 2:
            # Chọn ngẫu nhiên các cặp cùng class
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pairs_1.append(batch_data[indices[i]])
                    pairs_2.append(batch_data[indices[j]])
                    labels.append(0)  # 0 = same
    
    # Tạo negative pairs
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
# SIAMESE NETWORK
# ======================
class Siamese1DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward_once(self, x):
        return self.conv(x)

    def forward(self, x1, x2):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return F.pairwise_distance(e1, e2)

# ======================
# CONTRASTIVE LOSS
# ======================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, distances, labels):
        # labels: 0 = same, 1 = different
        loss_same = (1 - labels) * torch.pow(distances, 2)
        loss_diff = labels * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return torch.mean(loss_same + loss_diff)

# ======================
# TRAINING
# ======================
def train_model(train_loader, val_loader, epochs=50, lr=0.001, save_path='siamese_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Siamese1DNet().to(device)
    criterion = ContrastiveLoss(margin=1.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
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
            
            optimizer.zero_grad()
            distances = model(pairs_1, pairs_2)
            loss = criterion(distances, pair_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation
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
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved with val_loss: {avg_val_loss:.4f}")
    
    print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
    return model

# ======================
# MAIN
# ======================
def main():
    # Cấu trúc thư mục:
    # data/
    #   ├── type1/
    #   │   ├── img1.jpg
    #   │   ├── img2.jpg
    #   ├── type2/
    #   │   ├── img1.jpg
    #   │   ├── img2.jpg
    
    data_dir = "data"  # Thay đổi đường dẫn của bạn
    
    # Load dataset
    image_paths = []
    labels = []
    
    for label_idx, folder_name in enumerate(sorted(os.listdir(data_dir))):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, img_name))
                    labels.append(label_idx)
    
    print(f"Total images: {len(image_paths)}")
    print(f"Number of classes: {len(set(labels))}")
    
    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets với augmentation mạnh cho train
    train_dataset = ConnectorDataset(train_paths, train_labels, augment=True, aug_per_image=10)
    val_dataset = ConnectorDataset(val_paths, val_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train
    model = train_model(train_loader, val_loader, epochs=100, lr=0.001)
    
    print("\n✅ Training hoàn tất! Model đã được lưu vào 'siamese_model.pth'")

if __name__ == "__main__":
    main()