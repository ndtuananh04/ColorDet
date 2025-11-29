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
import json

# ======================
# ROI SELECTION & MANAGEMENT
# ======================
class ROIManager:
    def __init__(self, roi_file='roi_config.json'):
        self.roi_file = roi_file
        self.rois = self.load_rois()
        self.current_roi = None
        self.drawing = False
        self.start_point = None
        
    def load_rois(self):
        """Load ROI từ file JSON"""
        if os.path.exists(self.roi_file):
            with open(self.roi_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_rois(self):
        """Lưu ROI vào file JSON"""
        with open(self.roi_file, 'w') as f:
            json.dump(self.rois, f, indent=2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback để vẽ ROI bằng chuột"""
        image = param['image']
        window_name = param['window']
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = image.copy()
                cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_roi = (
                min(self.start_point[0], x),
                min(self.start_point[1], y),
                abs(x - self.start_point[0]),
                abs(y - self.start_point[1])
            )
            img_copy = image.copy()
            cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)
    
    def select_roi_interactive(self, image, window_name='Select ROI'):
        """Cho phép user vẽ ROI bằng chuột"""
        print("\n📍 Vẽ ROI bằng cách kéo chuột trên ảnh")
        print("   - Nhấn 'r' để reset")
        print("   - Nhấn 's' để save ROI")
        print("   - Nhấn 'q' để bỏ qua")
        
        cv2.namedWindow(window_name)
        param = {'image': image, 'window': window_name}
        cv2.setMouseCallback(window_name, self.mouse_callback, param)
        cv2.imshow(window_name, image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                self.current_roi = None
                cv2.imshow(window_name, image)
                print("ROI đã được reset")
            
            elif key == ord('s'):  # Save
                if self.current_roi is not None:
                    cv2.destroyWindow(window_name)
                    return self.current_roi
                else:
                    print("⚠️ Chưa vẽ ROI! Vui lòng vẽ trước khi save.")
            
            elif key == ord('q'):  # Quit without saving
                cv2.destroyWindow(window_name)
                return None
        
    def get_roi_for_image(self, image_path, image=None):
        """Lấy ROI cho một ảnh cụ thể"""
        # Kiểm tra xem đã có ROI cho ảnh này chưa
        if image_path in self.rois:
            roi = self.rois[image_path]
            return tuple(roi)  # (x, y, w, h)
        
        # Nếu chưa có, cho phép user chọn
        if image is None:
            image = cv2.imread(image_path)
        
        if image is None:
            return None
        
        print(f"\n🖼️ Chọn ROI cho: {os.path.basename(image_path)}")
        roi = self.select_roi_interactive(image, f'ROI - {os.path.basename(image_path)}')
        
        if roi is not None:
            self.rois[image_path] = list(roi)
            self.save_rois()
            print(f"✅ ROI đã được lưu: {roi}")
        
        return roi
    
    def set_default_roi(self, roi):
        """Đặt ROI mặc định cho tất cả ảnh"""
        self.rois['default'] = list(roi)
        self.save_rois()
    
    def get_default_roi(self):
        """Lấy ROI mặc định"""
        return tuple(self.rois.get('default', None)) if 'default' in self.rois else None

def crop_roi_region(image, roi):
    """Crop ảnh theo ROI đã định nghĩa"""
    if roi is None:
        return None
    
    x, y, w, h = roi
    # Đảm bảo ROI nằm trong ảnh
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None
    
    cropped = image[y:y+h, x:x+w]
    return cropped

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
    
    # Random brightness
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)
    
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
    """Preprocess ảnh đã crop thành input cho 1D CNN"""
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
    def __init__(self, image_paths, labels, roi_manager, augment=True, aug_per_image=5):
        """
        image_paths: list các đường dẫn ảnh
        labels: list các nhãn (connector type)
        roi_manager: ROIManager instance để lấy ROI
        augment: có áp dụng augmentation không
        aug_per_image: số lần augment mỗi ảnh
        """
        self.image_paths = image_paths
        self.labels = labels
        self.roi_manager = roi_manager
        self.augment = augment
        self.aug_per_image = aug_per_image
        
        # Load và crop tất cả ảnh theo ROI
        self.cropped_images = []
        self.valid_indices = []
        
        for idx, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is not None:
                # Lấy ROI cho ảnh này
                roi = self.roi_manager.get_roi_for_image(img_path, img)
                if roi is None:
                    # Thử dùng ROI mặc định
                    roi = self.roi_manager.get_default_roi()
                
                if roi is not None:
                    cropped = crop_roi_region(img, roi)
                    if cropped is not None and cropped.size > 0:
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
    """Tạo cặp positive và negative từ batch"""
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
# ROI SETUP HELPER
# ======================
def setup_rois_for_dataset(data_dir, roi_manager, use_default=True):
    """
    Hàm helper để setup ROI cho toàn bộ dataset
    
    Args:
        data_dir: thư mục chứa data
        roi_manager: ROIManager instance
        use_default: nếu True, chỉ cần chọn ROI một lần cho tất cả ảnh
    """
    if use_default:
        print("\n🎯 Chế độ ROI mặc định: chọn ROI một lần cho tất cả ảnh")
        
        # Lấy ảnh đầu tiên để chọn ROI
        sample_image_path = None
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        sample_image_path = os.path.join(folder_path, img_name)
                        break
                if sample_image_path:
                    break
        
        if sample_image_path:
            img = cv2.imread(sample_image_path)
            print(f"\n📷 Sử dụng ảnh mẫu: {os.path.basename(sample_image_path)}")
            roi = roi_manager.select_roi_interactive(img, 'Select Default ROI')
            if roi:
                roi_manager.set_default_roi(roi)
                print(f"✅ ROI mặc định đã được đặt: {roi}")
                return True
    else:
        print("\n🎯 Chế độ ROI riêng biệt: chọn ROI cho từng ảnh")
        print("   (ROI sẽ được chọn khi load dataset)")
    
    return False

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
    
    # Khởi tạo ROI Manager
    roi_manager = ROIManager(roi_file='roi_config.json')
    
    # Setup ROI (chọn một lần cho tất cả hoặc cho từng ảnh)
    setup_rois_for_dataset(data_dir, roi_manager, use_default=False)
    
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
    
    print(f"\n📊 Total images: {len(image_paths)}")
    print(f"📊 Number of classes: {len(set(labels))}")
    
    # Split train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets với augmentation
    train_dataset = ConnectorDataset(train_paths, train_labels, roi_manager, augment=True, aug_per_image=10)
    val_dataset = ConnectorDataset(val_paths, val_labels, roi_manager, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train
    model = train_model(train_loader, val_loader, epochs=100, lr=0.001)
    
    print("\n✅ Training hoàn tất! Model đã được lưu vào 'siamese_model.pth'")
    print(f"✅ ROI config đã được lưu vào '{roi_manager.roi_file}'")

if __name__ == "__main__":
    main()