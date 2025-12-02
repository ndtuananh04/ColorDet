import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ======================
# MODEL DEFINITION
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
# ROI SELECTOR FOR REAL-TIME
# ======================
class ROISelectorRealtime:
    """Class để di chuyển ROI cố định trong real-time camera"""
    def __init__(self, roi_width=30, roi_height=200):
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.roi_x = 0
        self.roi_y = 0
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Kiểm tra xem click có trong ROI không
            if (self.roi_x <= x <= self.roi_x + self.roi_width and 
                self.roi_y <= y <= self.roi_y + self.roi_height):
                self.dragging = True
                self.offset_x = x - self.roi_x
                self.offset_y = y - self.roi_y
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                # Di chuyển ROI theo chuột
                self.roi_x = x - self.offset_x
                self.roi_y = y - self.offset_y
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
    
    def get_roi(self):
        """Trả về ROI dạng (x, y, w, h)"""
        return (self.roi_x, self.roi_y, self.roi_width, self.roi_height)
    
    def clamp_roi(self, frame_width, frame_height):
        """Giới hạn ROI trong frame"""
        self.roi_x = max(0, min(self.roi_x, frame_width - self.roi_width))
        self.roi_y = max(0, min(self.roi_y, frame_height - self.roi_height))

def input_roi_size_realtime():
    """
    Nhập kích thước ROI cho real-time detection
    Returns: (width, height) tuple
    """
    print("\n" + "="*60)
    print("⚙️  CÀI ĐẶT KÍCH THƯỚC ROI")
    print("="*60)
    
    while True:
        try:
            print("\n📏 Nhập kích thước ROI (pixels):")
            print("   Gợi ý: 30x200 (dọc), 250x30 (ngang)")
            
            width_input = input("   Width (chiều rộng) [30]: ").strip()
            if not width_input:
                width = 30  # Default
                print(f"   → Sử dụng giá trị mặc định: {width}px")
            else:
                width = int(width_input)
            
            height_input = input("   Height (chiều cao) [200]: ").strip()
            if not height_input:
                height = 200  # Default
                print(f"   → Sử dụng giá trị mặc định: {height}px")
            else:
                height = int(height_input)
            
            # Kiểm tra giá trị hợp lệ
            if width <= 0 or height <= 0:
                print("❌ Kích thước phải lớn hơn 0!")
                continue
            
            if width > 1920 or height > 1080:
                print("⚠️  Cảnh báo: Kích thước quá lớn (tối đa 1920x1080)")
                confirm = input("   Tiếp tục? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            # Xác nhận
            print("\n" + "="*60)
            print(f"✅ Kích thước ROI: {width}x{height} pixels")
            print(f"   Diện tích: {width * height} pixels²")
            print("="*60)
            
            confirm = input("\nXác nhận kích thước này? (y/n): ").strip().lower()
            if confirm == 'y':
                return width, height
            else:
                print("🔄 Nhập lại...")
                
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã hủy!")
            return None, None

# ======================
# PREPROCESSING (PHẢI GIỐNG HỆT TRAINING)
# ======================
def preprocess_for_1dcnn(img, target_length=250):
    """
    Preprocess ảnh thành input cho 1D CNN
    CRITICAL: Hàm này PHẢI GIỐNG HỆT trong file training
    """
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
def load_trained_model(model_path='siamese_model.pth'):
    """Load model đã train"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Siamese1DNet().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Xử lý cả 2 trường hợp: save full checkpoint hoặc chỉ state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"   Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ Model loaded from: {model_path}")
        print(f"📱 Device: {device}")
        return model, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# ======================
# COMPARE TWO IMAGES
# ======================
def compare_two_images(model, device, img1_path, img2_path, threshold=0.5):
    """So sánh 2 ảnh (không cần ROI)"""
    # Đọc ảnh
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("❌ Cannot read images!")
        return None
    
    # Preprocess trực tiếp toàn bộ ảnh
    t1 = preprocess_for_1dcnn(img1)
    t2 = preprocess_for_1dcnn(img2)
    
    resized1 = cv2.resize(img1, (30, 200))
    resized2 = cv2.resize(img2, (30, 200))
    
    # 2. Chuyển sang HSV
    hsv1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2HSV)
    
    # 3. Normalize
    norm1 = hsv1.astype(np.float32) / 255.0
    norm2 = hsv2.astype(np.float32) / 255.0
    
    # 4. Average theo trục Y để tạo 1D signal
    avg_line1 = np.mean(norm1, axis=0)  # Shape: (250, 3)
    avg_line2 = np.mean(norm2, axis=0)
    
    # Tạo visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle('Preprocessing Visualization', fontsize=16, fontweight='bold')
    
    # Row 1: Original images
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Image 1: {os.path.basename(img1_path)}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Image 2: {os.path.basename(img2_path)}')
    axes[0, 1].axis('off')
    
    # Row 2: Resized (250x40)
    axes[1, 0].imshow(cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Resized (30x200)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(resized2, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Resized (30x200)')
    axes[1, 1].axis('off')
    
    # Row 3: HSV color space
    axes[2, 0].imshow(hsv1)
    axes[2, 0].set_title('HSV Color Space')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(hsv2)
    axes[2, 1].set_title('HSV Color Space')
    axes[2, 1].axis('off')
    
    # Row 4: 1D Signal (H, S, V channels)
    axes[3, 0].plot(avg_line1[:, 0], 'r-', label='H', linewidth=2)
    axes[3, 0].plot(avg_line1[:, 1], 'g-', label='S', linewidth=2)
    axes[3, 0].plot(avg_line1[:, 2], 'b-', label='V', linewidth=2)
    axes[3, 0].set_title('1D Signal (averaged along Y-axis)')
    axes[3, 0].set_xlabel('X position (0-250)')
    axes[3, 0].set_ylabel('Normalized value (0-1)')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    axes[3, 1].plot(avg_line2[:, 0], 'r-', label='H', linewidth=2)
    axes[3, 1].plot(avg_line2[:, 1], 'g-', label='S', linewidth=2)
    axes[3, 1].plot(avg_line2[:, 2], 'b-', label='V', linewidth=2)
    axes[3, 1].set_title('1D Signal (averaged along Y-axis)')
    axes[3, 1].set_xlabel('X position (0-250)')
    axes[3, 1].set_ylabel('Normalized value (0-1)')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    t1 = torch.tensor(t1).float().to(device)
    t2 = torch.tensor(t2).float().to(device)
    
    # Tính distance
    with torch.no_grad():
        distance = model(t1, t2).item()
    
    # Kết luận
    is_same = distance < threshold
    confidence = max(0, min(100, (1 - distance / (threshold * 2)) * 100))
    
    # In kết quả
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULT:")
    print(f"{'='*60}")
    print(f"Image 1: {os.path.basename(img1_path)}")
    print(f"Image 2: {os.path.basename(img2_path)}")
    print(f"Distance: {distance:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"{'='*60}")
    
    if is_same:
        print(f"✅ SAME CONNECTOR TYPE")
    else:
        print(f"❌ DIFFERENT CONNECTOR TYPE")
    
    print(f"{'='*60}\n")
    
    # Hiển thị ảnh
    display1 = cv2.resize(img1, (200, 300))
    display2 = cv2.resize(img2, (200, 300))
    combined = np.hstack([display1, display2])
    
    # Vẽ kết quả
    result_text = "SAME" if is_same else "DIFFERENT"
    result_color = (0, 255, 0) if is_same else (0, 0, 255)
    
    cv2.putText(combined, result_text, (combined.shape[1]//2 - 80, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3)
    cv2.putText(combined, f"Distance: {distance:.4f}", (20, combined.shape[0] - 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, f"Confidence: {confidence:.1f}%", (20, combined.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Comparison Result', combined)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return distance, is_same

# ======================
# REAL-TIME CAMERA WITH ROI
# ======================
def run_realtime_camera(model, device, reference_path, threshold=0.3, roi_width=30, roi_height=200, show_fps=True):
    """
    Chạy real-time với camera VÀ ROI có thể di chuyển
    - Crop ảnh từ ROI trên camera
    - So sánh với ảnh mẫu
    """
    # Load ảnh mẫu
    reference_img = cv2.imread(reference_path)
    if reference_img is None:
        print("❌ Cannot read reference image!")
        return
    
    # Preprocess ảnh mẫu
    reference_tensor_data = preprocess_for_1dcnn(reference_img)
    reference_tensor = torch.tensor(reference_tensor_data).float().to(device)
    
    print("\n" + "="*60)
    print("REAL-TIME CONNECTOR DETECTION WITH ROI")
    print("="*60)
    print("📷 Opening camera...")
    print(f"📋 Reference: {os.path.basename(reference_path)}")
    print(f"📏 ROI size: {roi_width}x{roi_height} pixels")
    print(f"🎯 Threshold: {threshold:.4f}")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current ROI frame")
    print("  '+' - Increase threshold by 0.05")
    print("  '-' - Decrease threshold by 0.05")
    print("  Drag ROI - Move ROI position")
    print("="*60 + "\n")
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    # Đọc frame đầu tiên để lấy kích thước
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read from camera!")
        cap.release()
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    
    # Kiểm tra ROI có vừa với frame không
    if roi_width > frame_width or roi_height > frame_height:
        print(f"❌ ROI ({roi_width}x{roi_height}) lớn hơn frame ({frame_width}x{frame_height})!")
        cap.release()
        return
    
    # Setup ROI selector
    roi_selector = ROISelectorRealtime(roi_width=roi_width, roi_height=roi_height)
    
    # Đặt ROI ở giữa màn hình
    roi_selector.roi_x = (frame_width - roi_width) // 2
    roi_selector.roi_y = (frame_height - roi_height) // 2
    
    window_name = 'Real-time Detection with ROI'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, roi_selector.mouse_callback)
    
    # FPS tracking
    import time
    prev_time = time.time()
    fps = 0
    
    print(f"🎯 ROI cố định: {roi_width}x{roi_height} pixels")
    print("📸 Kéo thả ROI để điều chỉnh vị trí...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        if show_fps:
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
        
        display = frame.copy()
        
        # Giới hạn ROI trong frame
        roi_selector.clamp_roi(frame_width, frame_height)
        
        # Lấy vị trí ROI hiện tại
        x, y, w, h = roi_selector.get_roi()
        
        try:
            # Crop ROI từ frame
            roi_frame = frame[y:y+h, x:x+w]
            if roi_frame.size == 0:
                raise ValueError("Empty ROI")
            
            # Preprocess ROI
            test_tensor_data = preprocess_for_1dcnn(roi_frame)
            test_tensor = torch.tensor(test_tensor_data).float().to(device)
            
            # Tính distance
            with torch.no_grad():
                distance = model(reference_tensor, test_tensor).item()
            
            is_match = distance < threshold
            
            # Vẽ ROI với màu tương ứng
            if roi_selector.dragging:
                roi_color = (0, 255, 255)  # Vàng khi đang kéo
            elif is_match:
                roi_color = (0, 255, 0)  # Xanh lá nếu match
            else:
                roi_color = (0, 0, 255)  # Đỏ nếu khác
            
            cv2.rectangle(display, (x, y), (x+w, y+h), roi_color, 3)
            
            # Vẽ tâm ROI
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(display, (center_x, center_y), 5, (255, 0, 255), -1)
            
            # Hiển thị kích thước ROI
            cv2.putText(display, f"ROI: {w}x{h}px", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
            
            # Hiển thị kết quả
            if is_match:
                status = "MATCH"
                status_color = (0, 255, 0)
                bar_color = (0, 255, 0)
            else:
                status = "DIFFERENT"
                status_color = (0, 0, 255)
                bar_color = (0, 0, 255)
            
            # Status text (góc trên trái)
            cv2.putText(display, status, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
            
            # Distance info
            cv2.putText(display, f"Distance: {distance:.4f}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Threshold: {threshold:.4f}", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Confidence bar
            confidence = max(0, min(100, (1 - distance / (threshold * 2)) * 100))
            bar_width = int(confidence * 3)  # 300px max
            cv2.rectangle(display, (20, 150), (20 + bar_width, 180), bar_color, -1)
            cv2.rectangle(display, (20, 150), (320, 180), (255, 255, 255), 2)
            cv2.putText(display, f"{confidence:.1f}%", (330, 175),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # FPS
            if show_fps:
                cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Hiển thị ROI preview (nhỏ ở góc phải dưới)
            preview_size = 150
            roi_resized = cv2.resize(roi_frame, (preview_size, int(preview_size * h / w)))
            preview_y = display.shape[0] - roi_resized.shape[0] - 10
            preview_x = display.shape[1] - preview_size - 10
            display[preview_y:preview_y+roi_resized.shape[0], 
                   preview_x:preview_x+preview_size] = roi_resized
            cv2.rectangle(display, (preview_x, preview_y), 
                         (preview_x+preview_size, preview_y+roi_resized.shape[0]), 
                         (255, 255, 255), 2)
            cv2.putText(display, "ROI Preview", (preview_x, preview_y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            cv2.putText(display, f"Error: {str(e)[:50]}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Vẽ ROI màu đỏ khi có lỗi
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Controls help (góc dưới trái)
        help_y = display.shape[0] - 70
        cv2.putText(display, "Drag ROI to move | q: Quit | s: Save | +/-: Threshold", 
                   (20, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('s'):
            # Lưu ảnh ROI
            try:
                roi_frame = frame[y:y+h, x:x+w]
                filename = f'roi_captured_{int(time.time())}.jpg'
                cv2.imwrite(filename, roi_frame)
                print(f"✅ Saved ROI: {filename}")
            except:
                print(f"❌ Cannot save ROI!")
        
        elif key == ord('+') or key == ord('='):
            threshold += 0.05
            print(f"Threshold: {threshold:.4f}")
        
        elif key == ord('-') or key == ord('_'):
            threshold = max(0.05, threshold - 0.05)
            print(f"Threshold: {threshold:.4f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera closed")

# ======================
# BATCH COMPARISON
# ======================
def batch_compare_with_reference(model, device, reference_path, test_folder, threshold=0.5):
    """So sánh tất cả ảnh trong folder với ảnh mẫu"""
    
    # Load reference image
    reference_img = cv2.imread(reference_path)
    if reference_img is None:
        print("❌ Cannot read reference image!")
        return
    
    reference_tensor_data = preprocess_for_1dcnn(reference_img)
    reference_tensor = torch.tensor(reference_tensor_data).float().to(device)
    
    # Get all images in folder
    image_files = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    if len(image_files) == 0:
        print("❌ No images found in folder!")
        return
    
    print("\n" + "="*60)
    print(f"BATCH COMPARISON")
    print("="*60)
    print(f"Reference: {os.path.basename(reference_path)}")
    print(f"Test folder: {test_folder}")
    print(f"Total images: {len(image_files)}")
    print(f"Threshold: {threshold:.4f}")
    print("="*60 + "\n")
    
    results = []
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_folder, img_file)
        test_img = cv2.imread(img_path)
        
        if test_img is None:
            print(f"[{idx}/{len(image_files)}] ⚠️ Cannot read: {img_file}")
            continue
        
        # Preprocess and compare
        test_tensor_data = preprocess_for_1dcnn(test_img)
        test_tensor = torch.tensor(test_tensor_data).float().to(device)
        
        with torch.no_grad():
            distance = model(reference_tensor, test_tensor).item()
        
        is_match = distance < threshold
        confidence = max(0, min(100, (1 - distance / (threshold * 2)) * 100))
        
        status = "✅ MATCH" if is_match else "❌ DIFFERENT"
        print(f"[{idx}/{len(image_files)}] {img_file:30s} | Distance: {distance:.4f} | {status}")
        
        results.append({
            'filename': img_file,
            'distance': distance,
            'is_match': is_match,
            'confidence': confidence
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total = len(results)
    matched = sum(1 for r in results if r['is_match'])
    print(f"Total processed: {total}")
    print(f"Matched: {matched} ({matched/total*100:.1f}%)")
    print(f"Different: {total - matched} ({(total-matched)/total*100:.1f}%)")
    print("="*60 + "\n")
    
    return results

# ======================
# MAIN MENU
# ======================
def main():
    print("\n" + "="*60)
    print("SIAMESE NETWORK - CONNECTOR DETECTION")
    print("="*60)
    
    # Load model
    model_path = input("Model path (Enter = siamese_model.pth): ").strip()
    if not model_path:
        model_path = 'siamese_model.pth'
    
    model, device = load_trained_model(model_path)
    if model is None:
        return
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Compare two images")
        print("2. Real-time camera detection (with ROI)")
        print("3. Batch comparison (folder)")
        print("4. Exit")
        print("="*60)
        
        choice = input("Choose option (1/2/3/4): ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            print("COMPARE TWO IMAGES")
            print("="*60)
            img1 = input("Image 1 path: ").strip()
            img2 = input("Image 2 path: ").strip()
            threshold = float(input("Threshold (Enter = 0.5): ").strip() or "0.5")
            
            compare_two_images(model, device, img1, img2, threshold)
        
        elif choice == "2":
            print("\n" + "="*60)
            print("REAL-TIME CAMERA DETECTION WITH ROI")
            print("="*60)
            reference = input("Reference image path: ").strip()
            
            # Nhập kích thước ROI
            roi_width, roi_height = input_roi_size_realtime()
            if roi_width is None or roi_height is None:
                print("❌ Đã hủy!")
                continue
            
            threshold = float(input("Threshold (Enter = 0.3): ").strip() or "0.3")
            
            run_realtime_camera(model, device, reference, threshold, roi_width, roi_height)
        
        elif choice == "3":
            print("\n" + "="*60)
            print("BATCH COMPARISON")
            print("="*60)
            reference = input("Reference image path: ").strip()
            test_folder = input("Test folder path: ").strip()
            threshold = float(input("Threshold (Enter = 0.5): ").strip() or "0.5")
            
            batch_compare_with_reference(model, device, reference, test_folder, threshold)
        
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("⚠️ Invalid choice!")

if __name__ == "__main__":
    main()