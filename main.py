import cv2
import numpy as np

def detect_metal_connectors(frame):
    """
    Phát hiện các chấu kim loại trong đầu nối và trả về bounding box 
    bao gồm cả 30px dây cắm vào
    """
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng blur để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological operations để kết nối các cạnh gần nhau
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Tìm contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tìm tất cả các chấu kim loại
    metal_pins = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Lọc theo diện tích (các chấu kim loại nhỏ)
        if area > 50 and area < 2000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Lọc theo tỷ lệ khung hình
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Chấu kim loại thường có tỷ lệ gần vuông
            if 0.3 < aspect_ratio < 3.0:
                metal_pins.append((x, y, w, h))
    
    # Nếu không tìm thấy chấu kim loại nào
    if len(metal_pins) == 0:
        return None
    
    # Tìm bounding box bao quanh tất cả các chấu kim loại
    all_x = [x for x, y, w, h in metal_pins]
    all_y = [y for x, y, w, h in metal_pins]
    all_x_max = [x + w for x, y, w, h in metal_pins]
    all_y_max = [y + h for x, y, w, h in metal_pins]
    
    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max(all_x_max)
    max_y = max(all_y_max)
    
    # Mở rộng bounding box về phía dây cắm (giả sử dây ở bên trái)
    # Thêm 60px về phía trái
    extension = 80
    min_x = max(0, min_x - extension)
    max_x = min_x + 30
    # Có thể thêm một chút padding cho chiều cao
    padding = 20
    min_y = max(0, min_y - padding)
    max_y = min(frame.shape[0], max_y + padding)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return (min_x, min_y, width, height), metal_pins

def main():
    # Đọc ảnh từ file
    image_path = "4.jpg"  # Thay đổi đường dẫn ảnh của bạn
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Không thể đọc ảnh từ: {image_path}")
        print("Vui lòng kiểm tra đường dẫn file!")
        return
    
    print("Đang xử lý ảnh...")
    
    # Phát hiện các đầu nối kim loại
    result_detection = detect_metal_connectors(frame)
    
    # Kiểm tra nếu không tìm thấy
    if result_detection is None:
        print("Không phát hiện được đầu nối kim loại!")
        return
    
    # Unpack kết quả: bounding box chính và các chấu kim loại
    main_bbox, metal_pins = result_detection
    
    # Tạo bản sao để vẽ
    result = frame.copy()
    
    # Vẽ bounding box chính (bao gồm cả dây cắm) - màu xanh lá đậm
    x, y, w, h = main_bbox
    x, y, w, h = int(x), int(y), int(w), int(h)  # Ép kiểu về int
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(result, "Main Connector", (x, y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Vẽ các chấu kim loại riêng lẻ (màu xanh dương để phân biệt)
    for (px, py, pw, ph) in metal_pins:
        px, py, pw, ph = int(px), int(py), int(pw), int(ph)
        cv2.rectangle(result, (px, py), (px + pw, py + ph), (255, 0, 0), 1)
    
    # Hiển thị số lượng phát hiện
    cv2.putText(result, f"Metal pins: {len(metal_pins)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print(f"Đã phát hiện {len(metal_pins)} chấu kim loại")
    print(f"Bounding box chính: x={x}, y={y}, w={w}, h={h}")
    
    # Lưu ảnh kết quả
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"Đã lưu ảnh kết quả: {output_path}")
    
    # Hiển thị ảnh
    cv2.imshow('Original Image', frame)
    cv2.imshow('Detection Result', result)
    
    print("Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()