import cv2
import numpy as np

def detect_metal_connectors(frame):
    """Detect các chấu kim loại của connector"""
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
        if area > 50 and area < 2000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.3 < aspect_ratio < 3.0:
                metal_pins.append((x, y, w, h))
    
    if len(metal_pins) == 0:
        return None
    
    metal_pins.sort(key=lambda pin: pin[1])
    return metal_pins

def get_wire_region(frame, metal_pins):
    """Xác định vùng chứa các dây (bên trái connector)"""
    if not metal_pins:
        return None
    
    # Tìm vị trí trái nhất của các pin
    leftmost_x = min([x for x, y, w, h in metal_pins])
    
    # Vùng dây: từ trái ảnh đến trước connector
    wire_region_x = 0
    wire_region_width = leftmost_x - 10
    
    # Chiều cao: từ pin đầu tiên đến pin cuối cùng + buffer
    top_y = min([y for x, y, w, h in metal_pins]) - 20
    bottom_y = max([y + h for x, y, w, h in metal_pins]) + 20
    
    # Đảm bảo không vượt biên
    top_y = max(0, top_y)
    bottom_y = min(frame.shape[0], bottom_y)
    wire_region_width = max(10, wire_region_width)
    
    return (wire_region_x, top_y, wire_region_width, bottom_y - top_y)

def detect_individual_wires(frame, wire_region):
    """Detect từng dây riêng lẻ bằng color segmentation"""
    x, y, w, h = wire_region
    roi = frame[y:y+h, x:x+w]
    
    # Chuyển sang HSV để dễ phân tách màu
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Tạo mask loại bỏ nền (đen/trắng/xám)
    # Lọc theo saturation (độ bão hòa màu)
    lower_color = np.array([0, 30, 30])  # Loại bỏ màu nhạt
    upper_color = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Morphological operations để làm mịn
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Tìm contours của từng dây
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wires = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Lọc theo kích thước (dây thường có diện tích nhất định)
        if area > 100 and area < 5000:
            # Lấy bounding box
            wx, wy, ww, wh = cv2.boundingRect(contour)
            
            # Lọc theo tỷ lệ (dây thường dài hơn rộng)
            aspect_ratio = float(ww) / wh if wh > 0 else 0
            
            # Dây có thể ngang hoặc hơi chéo
            if ww > 10 and wh > 5:  # Đảm bảo có kích thước tối thiểu
                # Chuyển tọa độ về frame gốc
                global_x = x + wx
                global_y = y + wy
                
                wires.append({
                    'bbox': (global_x, global_y, ww, wh),
                    'contour': contour,
                    'area': area,
                    'center_y': global_y + wh // 2
                })
    
    # Sắp xếp các dây từ trên xuống dưới
    wires.sort(key=lambda w: w['center_y'])
    
    return wires

def get_color_name(bgr_color):
    """Xác định tên màu từ giá trị BGR"""
    b, g, r = bgr_color
    
    colors = {
        'Red': ([0, 0, 150], [80, 80, 255]),
        'Yellow': ([0, 150, 150], [100, 255, 255]),
        'Green': ([0, 100, 0], [120, 255, 120]),
        'Blue': ([100, 0, 0], [255, 100, 100]),
        'Black': ([0, 0, 0], [60, 60, 60]),
        'White': ([200, 200, 200], [255, 255, 255]),
        'Brown': ([0, 30, 80], [100, 120, 200]),
        'Orange': ([0, 100, 200], [100, 200, 255]),
        'Gray': ([70, 70, 70], [180, 180, 180]),
        'Purple': ([100, 0, 80], [255, 80, 180]),
    }
    
    for color_name, (lower, upper) in colors.items():
        if (lower[0] <= b <= upper[0] and 
            lower[1] <= g <= upper[1] and 
            lower[2] <= r <= upper[2]):
            return color_name
    
    return 'Unknown'

def get_dominant_color(roi):
    """Lấy màu chủ đạo từ ROI bằng K-means"""
    pixels = roi.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Lọc pixel theo độ sáng
    brightness = np.mean(pixels, axis=1)
    mask = (brightness > 30) & (brightness < 220)
    
    if np.sum(mask) < 10:
        filtered_pixels = pixels
    else:
        filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) == 0:
        return (0, 0, 0)
    
    # K-means clustering
    n_clusters = min(3, len(filtered_pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(filtered_pixels, n_clusters, None, 
                                     criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    label_counts = np.bincount(labels.flatten())
    dominant_idx = np.argmax(label_counts)
    dominant_color = centers[dominant_idx]
    
    return tuple(map(int, dominant_color))

def detect_wire_colors(frame, wires):
    """Xác định màu của từng dây"""
    wire_colors = []
    
    for idx, wire in enumerate(wires):
        wx, wy, ww, wh = wire['bbox']
        
        # Lấy vùng ROI của dây (phần giữa của bounding box)
        # Lấy 70% giữa để tránh viền
        margin_x = int(ww * 0.15)
        margin_y = int(wh * 0.15)
        
        sample_x = wx + margin_x
        sample_y = wy + margin_y
        sample_w = max(1, ww - 2 * margin_x)
        sample_h = max(1, wh - 2 * margin_y)
        
        # Đảm bảo không vượt biên
        sample_x = max(0, sample_x)
        sample_y = max(0, sample_y)
        sample_w = min(sample_w, frame.shape[1] - sample_x)
        sample_h = min(sample_h, frame.shape[0] - sample_y)
        
        roi = frame[sample_y:sample_y+sample_h, sample_x:sample_x+sample_w]
        
        if roi.size == 0 or sample_w < 1 or sample_h < 1:
            wire_colors.append({
                'index': idx + 1,
                'color': 'Unknown',
                'bgr': (0, 0, 0),
                'bbox': (wx, wy, ww, wh)
            })
            continue
        
        # Lấy màu chủ đạo
        dominant_color = get_dominant_color(roi)
        color_name = get_color_name(dominant_color)
        
        wire_colors.append({
            'index': idx + 1,
            'color': color_name,
            'bgr': dominant_color,
            'bbox': (wx, wy, ww, wh)
        })
    
    return wire_colors

def main():
    # Đọc ảnh
    image_path = "nho.jpg"
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Không thể đọc ảnh từ: {image_path}")
        return
    
    print("Đang phát hiện connector và dây...")
    
    # 1. Detect chấu kim loại
    metal_pins = detect_metal_connectors(frame)
    
    if metal_pins is None:
        print("Không tìm thấy chấu kim loại!")
        return
    
    print(f"Tìm thấy {len(metal_pins)} chấu kim loại")
    
    # 2. Xác định vùng chứa dây
    wire_region = get_wire_region(frame, metal_pins)
    
    # 3. Detect từng dây riêng lẻ
    wires = detect_individual_wires(frame, wire_region)
    print(f"Tìm thấy {len(wires)} dây")
    
    # 4. Xác định màu của từng dây
    wire_colors = detect_wire_colors(frame, wires)
    
    # Vẽ kết quả
    result_img = frame.copy()
    
    # Vẽ vùng chứa dây
    x, y, w, h = wire_region
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (128, 128, 128), 1)
    
    # Vẽ từng dây
    for i, wire_info in enumerate(wire_colors):
        wx, wy, ww, wh = wire_info['bbox']
        
        # Vẽ bounding box của dây
        cv2.rectangle(result_img, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 2)
        
        # Vẽ số thứ tự và tên màu
        label = f"{i+1}. {wire_info['color']}"
        label_y = wy - 5 if wy > 20 else wy + wh + 15
        
        # Vẽ background cho text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result_img, (wx, label_y - text_h - 2),
                     (wx + text_w, label_y + 2), (0, 0, 0), -1)
        
        # Vẽ text
        cv2.putText(result_img, label, (wx, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Vẽ điểm màu mẫu
        color_sample_x = wx + ww + 10
        color_sample_y = wy + wh // 2
        cv2.circle(result_img, (color_sample_x, color_sample_y), 8, 
                  wire_info['bgr'], -1)
        cv2.circle(result_img, (color_sample_x, color_sample_y), 8, 
                  (0, 0, 0), 2)
    
    # In kết quả
    print(f"\n{'='*50}")
    print(f"KẾT QUẢ PHÁT HIỆN MÀU DÂY")
    print(f"{'='*50}")
    print(f"Tổng số dây: {len(wire_colors)}")
    print(f"Thứ tự từ TRÊN xuống DƯỚI:")
    print(f"{'-'*50}")
    
    for wire_info in wire_colors:
        bgr = wire_info['bgr']
        print(f"  {wire_info['index']}. {wire_info['color']:12s} - BGR: {bgr}")
    
    print(f"{'='*50}\n")
    
    # Lưu và hiển thị
    output_path = "wire_detection_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"Đã lưu ảnh kết quả: {output_path}")
    
    cv2.imshow('Original', frame)
    cv2.imshow('Wire Detection', result_img)
    
    print("Nhấn phím bất kỳ để đóng...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()