import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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

    # Mở rộng về phía dây (bên trái)
    extension = 80
    min_x = max(0, min_x - extension)
    max_x = min_x + 30
    padding = 20
    min_y = max(0, min_y - padding)
    max_y = min(frame.shape[0], max_y + padding)

    width = max_x - min_x
    height = max_y - min_y

    return (min_x, min_y, width, height), metal_pins

# ======================
# CẮT VÙNG CHÍNH
# ======================
def crop_main_region(image):
    result = detect_metal_connectors(image)
    if result is None:
        return None
    (x, y, w, h), _ = result
    cropped = image[y:y+h, x:x+w]
    return cropped

# ======================
# TIỀN XỬ LÝ CHO 1D CNN
# ======================
def preprocess_for_1dcnn(cropped, target_length=250):
    # Resize về 40x250
    resized = cv2.resize(cropped, (target_length, 40))

    # ✅ Chuyển sang không gian HSV
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Chuẩn hóa về [0, 1]
    norm = hsv.astype(np.float32) / 255.0

    # Trung bình theo chiều cao → (40, 250, 3) → (250, 3)
    avg_line = np.mean(norm, axis=0)

    # Hoán đổi trục → (3, 250)
    line_1d = avg_line.T

    # Thêm batch dimension → (1, 3, 250)
    return np.expand_dims(line_1d, axis=0)

# ======================
# MẠNG SIAMESE 1D
# ======================
class Siamese1DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.ReLU()
        )

    def forward_once(self, x):
        return self.conv(x)

    def forward(self, x1, x2):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return F.pairwise_distance(e1, e2)

# ======================
# HÀM SO SÁNH 2 ẢNH
# ======================
def compare_connectors(model, frame1, frame2, threshold=0.7):
    crop1 = crop_main_region(frame1)
    crop2 = crop_main_region(frame2)

    if crop1 is None or crop2 is None:
        print("Không phát hiện được vùng đầu nối trong một trong hai ảnh.")
        return None
    cv2.imshow("Crop1", crop2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    t1 = preprocess_for_1dcnn(crop1)
    t2 = preprocess_for_1dcnn(crop2)

    t1 = torch.tensor(t1).float()
    t2 = torch.tensor(t2).float()

    with torch.no_grad():
        dist = model(t1, t2).item()

    print(f"Khoảng cách giữa hai đầu nối: {dist:.4f}")
    if dist < threshold:
        print("✅ Hai đầu nối có vẻ CÙNG LOẠI.")
    else:
        print("❌ Hai đầu nối KHÁC LOẠI.")
    return dist

# ======================
# MAIN
# ======================
def main():
    image_path1 = "4.jpg"
    image_path2 = "6.jpg"

    frame1 = cv2.imread(image_path1)
    frame2 = cv2.imread(image_path2)

    if frame1 is None or frame2 is None:
        print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn!")
        return

    print("Đang phát hiện vùng đầu nối và so sánh...")

    model = Siamese1DNet()
    model.eval()  # chưa huấn luyện, chỉ chạy pipeline thử

    distance = compare_connectors(model, frame1, frame2)
    time.sleep(10)
    if distance is not None:
        print(f"Hoàn tất. Khoảng cách đặc trưng: {distance:.4f}")

if __name__ == "__main__":
    main()
