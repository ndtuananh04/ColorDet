import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# ĐỊNH NGHĨA MODEL (giống code training)
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

    extension = 80
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

def preprocess_for_1dcnn(cropped, target_length=250):
    resized = cv2.resize(cropped, (target_length, 40))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    norm = hsv.astype(np.float32) / 255.0
    avg_line = np.mean(norm, axis=0)
    line_1d = avg_line.T
    return np.expand_dims(line_1d, axis=0)

# ======================
# LOAD MODEL
# ======================
def load_trained_model(model_path='siamese_model.pth'):
    """Load model đã train"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Siamese1DNet().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Đã load model từ: {model_path}")
        print(f"Device: {device}")
        return model, device
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None, None

# ======================
# SO SÁNH 2 ẢNH
# ======================
def compare_two_images(model, device, img1_path, img2_path, threshold=0.5):
    """
    So sánh 2 ảnh connector
    threshold: ngưỡng quyết định (distance < threshold → giống nhau)
    """
    # Đọc ảnh
    frame1 = cv2.imread(img1_path)
    frame2 = cv2.imread(img2_path)
    
    if frame1 is None or frame2 is None:
        print("❌ Không thể đọc ảnh!")
        return None
    
    # Crop vùng đầu nối
    crop1 = crop_main_region(frame1)
    crop2 = crop_main_region(frame2)
    
    if crop1 is None or crop2 is None:
        print("❌ Không phát hiện được vùng đầu nối trong một trong hai ảnh")
        return None
    
    # Preprocess
    t1 = preprocess_for_1dcnn(crop1)
    t2 = preprocess_for_1dcnn(crop2)
    
    t1 = torch.tensor(t1).float().to(device)
    t2 = torch.tensor(t2).float().to(device)
    
    # Tính distance
    with torch.no_grad():
        distance = model(t1, t2).item()
    
    # Kết luận
    is_same = distance < threshold
    confidence = 1 - (distance / (threshold * 2))  # Độ tin cậy
    confidence = max(0, min(1, confidence)) * 100
    
    print(f"\n{'='*60}")
    print(f"SO SÁNH KẾT QUẢ:")
    print(f"{'='*60}")
    print(f"Distance: {distance:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"{'='*60}")
    
    if is_same:
        print(f"✅ HAI ĐẦU NỐI GIỐNG NHAU (cùng thứ tự màu dây)")
    else:
        print(f"❌ HAI ĐẦU NỐI KHÁC NHAU (thứ tự màu dây khác)")
    
    print(f"{'='*60}\n")
    
    # Hiển thị ảnh
    display1 = frame1.copy()
    display2 = frame2.copy()
    
    # Vẽ crop region
    result1 = detect_metal_connectors(frame1)
    result2 = detect_metal_connectors(frame2)
    
    if result1 is not None:
        (x, y, w, h), _ = result1
        cv2.rectangle(display1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if result2 is not None:
        (x, y, w, h), _ = result2
        cv2.rectangle(display2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Resize để hiển thị
    h1, w1 = display1.shape[:2]
    h2, w2 = display2.shape[:2]
    max_h = max(h1, h2)
    
    display1 = cv2.resize(display1, (int(w1*max_h/h1), max_h))
    display2 = cv2.resize(display2, (int(w2*max_h/h2), max_h))
    
    combined = np.hstack([display1, display2])
    
    # Vẽ kết quả lên ảnh
    result_text = "SAME" if is_same else "DIFFERENT"
    result_color = (0, 255, 0) if is_same else (0, 0, 255)
    cv2.putText(combined, result_text, (combined.shape[1]//2 - 80, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3)
    cv2.putText(combined, f"Distance: {distance:.4f}", (combined.shape[1]//2 - 120, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Comparison Result', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return distance, is_same

# ======================
# SO SÁNH VỚI MẪU CHUẨN
# ======================
class ReferenceChecker:
    """Lưu mẫu chuẩn và so sánh real-time"""
    def __init__(self, model, device, threshold=0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.reference_tensor = None
        self.reference_image = None
        self.is_set = False
    
    def set_reference(self, image_path):
        """Lưu ảnh làm mẫu chuẩn"""
        frame = cv2.imread(image_path)
        if frame is None:
            return False, "Không thể đọc ảnh"
        
        crop = crop_main_region(frame)
        if crop is None:
            return False, "Không phát hiện được đầu nối"
        
        tensor = preprocess_for_1dcnn(crop)
        self.reference_tensor = torch.tensor(tensor).float().to(self.device)
        self.reference_image = frame.copy()
        self.is_set = True
        
        return True, "Đã lưu mẫu chuẩn"
    
    def check(self, image_path):
        """So sánh ảnh với mẫu chuẩn"""
        if not self.is_set:
            return None, "Chưa có mẫu chuẩn"
        
        frame = cv2.imread(image_path)
        if frame is None:
            return None, "Không thể đọc ảnh"
        
        crop = crop_main_region(frame)
        if crop is None:
            return None, "Không phát hiện được đầu nối"
        
        tensor = preprocess_for_1dcnn(crop)
        test_tensor = torch.tensor(tensor).float().to(self.device)
        
        with torch.no_grad():
            distance = self.model(self.reference_tensor, test_tensor).item()
        
        is_match = distance < self.threshold
        return distance, is_match

# ======================
# REAL-TIME CAMERA
# ======================
def run_realtime_checking(model, device, reference_path, threshold=0.5):
    """
    Chạy real-time với camera, so sánh với mẫu chuẩn
    """
    checker = ReferenceChecker(model, device, threshold)
    success, msg = checker.set_reference(reference_path)
    model = model
    if not success:
        print(f"❌ {msg}")
        return
    
    print(f"✅ {msg}")
    print("\n" + "="*60)
    print("REAL-TIME CHECKING")
    print("="*60)
    print("Nhấn 'q' để thoát")
    print("Nhấn 's' để lưu ảnh hiện tại")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Crop và kiểm tra
        crop = crop_main_region(frame)
        
        if crop is not None:
            # Vẽ vùng detect
            result = detect_metal_connectors(frame)
            if result is not None:
                (x, y, w, h), _ = result
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # So sánh với mẫu
            tensor = preprocess_for_1dcnn(crop)
            test_tensor = torch.tensor(tensor).float().to(device)
            
            with torch.no_grad():
                distance = model(checker.reference_tensor, test_tensor).item()
            
            is_match = distance < threshold
            
            # Hiển thị kết quả
            if is_match:
                status = "MATCH"
                color = (0, 255, 0)
            else:
                status = "DIFFERENT"
                color = (0, 0, 255)
            
            cv2.putText(display, status, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display, f"Distance: {distance:.4f}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Threshold: {threshold:.4f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display, "No Connector Detected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Real-time Checking', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('captured.jpg', frame)
            print("✅ Đã lưu ảnh: captured.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

# ======================
# MAIN
# ======================
def main():
    import sys
    
    # Load model
    model, device = load_trained_model('siamese_model2.pth')
    if model is None:
        return
    
    print("\n" + "="*60)
    print("CHỌN CHỨC NĂNG:")
    print("="*60)
    print("1. So sánh 2 ảnh")
    print("2. So sánh với mẫu chuẩn (real-time camera)")
    print("3. So sánh với mẫu chuẩn (từ file)")
    print("="*60)
    
    choice = input("Nhập lựa chọn (1/2/3): ").strip()
    
    if choice == "1":
        # So sánh 2 ảnh
        print("\nSo sánh 2 ảnh")
        img1 = input("Đường dẫn ảnh 1: ").strip()
        img2 = input("Đường dẫn ảnh 2: ").strip()
        threshold = float(input("Threshold (mặc định 0.5): ").strip() or "0.5")
        
        compare_two_images(model, device, img1, img2, threshold)
    
    elif choice == "2":
        # Real-time với camera
        print("\nReal-time checking")
        reference = input("Đường dẫn ảnh mẫu chuẩn: ").strip()
        threshold = float(input("Threshold (mặc định 0.5): ").strip() or "0.5")
        
        run_realtime_checking(model, device, reference, threshold)
    
    elif choice == "3":
        # So sánh từ file với mẫu
        print("\nSo sánh với mẫu chuẩn")
        reference = input("Đường dẫn ảnh mẫu chuẩn: ").strip()
        test_img = input("Đường dẫn ảnh cần kiểm tra: ").strip()
        threshold = float(input("Threshold (mặc định 0.5): ").strip() or "0.5")
        
        checker = ReferenceChecker(model, device, threshold)
        success, msg = checker.set_reference(reference)
        
        if success:
            print(f"✅ {msg}")
            distance, is_match = checker.check(test_img)
            
            if distance is not None:
                print(f"\nDistance: {distance:.4f}")
                if is_match:
                    print("✅ KHỚP VỚI MẪU CHUẨN")
                else:
                    print("❌ KHÔNG KHỚP VỚI MẪU CHUẨN")
        else:
            print(f"❌ {msg}")
    
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()