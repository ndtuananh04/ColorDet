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
# PREPROCESSING (PHẢI GIỐNG HỆT TRAINING)
# ======================
def preprocess_for_1dcnn(img, target_length=250):
    """
    Preprocess ảnh thành input cho 1D CNN
    CRITICAL: Hàm này PHẢI GIỐNG HỆT trong file training
    """
    # Resize về kích thước chuẩn (width=target_length, height=40)
    resized = cv2.resize(img, (target_length, 40))
    
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
    display1 = cv2.resize(img1, (400, 300))
    display2 = cv2.resize(img2, (400, 300))
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
# REAL-TIME CAMERA
# ======================
def run_realtime_camera(model, device, reference_path, threshold=0.3, show_fps=True):
    """
    Chạy real-time với camera (không cần ROI)
    - So sánh toàn bộ khung hình với ảnh mẫu
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
    print("REAL-TIME CONNECTOR DETECTION")
    print("="*60)
    print("📷 Opening camera...")
    print(f"📋 Reference: {os.path.basename(reference_path)}")
    print(f"🎯 Threshold: {threshold:.4f}")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  '+' - Increase threshold by 0.05")
    print("  '-' - Decrease threshold by 0.05")
    print("="*60 + "\n")
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    # FPS tracking
    import time
    prev_time = time.time()
    fps = 0
    
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
        
        try:
            # Preprocess frame hiện tại
            test_tensor_data = preprocess_for_1dcnn(frame)
            test_tensor = torch.tensor(test_tensor_data).float().to(device)
            
            # Tính distance
            with torch.no_grad():
                distance = model(reference_tensor, test_tensor).item()
            
            is_match = distance < threshold
            
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
            
            # Controls help (góc dưới)
            help_y = display.shape[0] - 70
            cv2.putText(display, "q: Quit | s: Save | +/-: Adjust threshold", (20, help_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
        except Exception as e:
            cv2.putText(display, f"Error: {str(e)[:50]}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Real-time Detection', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('s'):
            filename = f'captured_{int(time.time())}.jpg'
            cv2.imwrite(filename, frame)
            print(f"✅ Saved: {filename}")
        
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
        print("2. Real-time camera detection")
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
            print("REAL-TIME CAMERA DETECTION")
            print("="*60)
            reference = input("Reference image path: ").strip()
            threshold = float(input("Threshold (Enter = 0.3): ").strip() or "0.3")
            
            run_realtime_camera(model, device, reference, threshold)
        
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