import cv2
import numpy as np

def detect_metal_connectors(frame):
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
    
    # Sắp xếp các chấu kim loại từ trên xuống dưới (theo tọa độ y)
    metal_pins.sort(key=lambda pin: pin[1])
    
    return metal_pins

def get_color_name(bgr_color):
    """
    Xác định tên màu từ giá trị BGR
    """
    b, g, r = bgr_color
    
    # Định nghĩa các màu chuẩn (BGR format)
    colors = {
        'Red': ([0, 0, 150], [80, 80, 255]),
        'Yellow': ([0, 150, 150], [80, 255, 255]),
        'Green': ([0, 100, 0], [100, 255, 100]),
        'Blue': ([100, 0, 0], [255, 100, 100]),
        'Black': ([0, 0, 0], [50, 50, 50]),
        'White': ([200, 200, 200], [255, 255, 255]),
        'Brown': ([0, 40, 100], [80, 100, 200]),
        'Orange': ([0, 100, 200], [80, 180, 255]),
        'Gray': ([80, 80, 80], [180, 180, 180]),
    }
    
    # Tìm màu gần nhất
    for color_name, (lower, upper) in colors.items():
        if (lower[0] <= b <= upper[0] and 
            lower[1] <= g <= upper[1] and 
            lower[2] <= r <= upper[2]):
            return color_name
    
    return 'Unknown'

def get_dominant_color(roi):
    """
    Lấy màu chủ đạo từ vùng ROI bằng K-means clustering
    Loại bỏ màu nền (đen/trắng)
    """
    # Reshape ảnh thành mảng 2D của pixels
    pixels = roi.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Loại bỏ các pixel quá tối (đen) hoặc quá sáng (trắng)
    # Lọc ra các pixel có độ sáng từ 30 đến 220
    brightness = np.mean(pixels, axis=1)
    mask = (brightness > 30) & (brightness < 220)
    
    if np.sum(mask) < 10:  # Nếu quá ít pixel hợp lệ
        # Nếu không đủ pixel màu, lấy tất cả
        filtered_pixels = pixels
    else:
        filtered_pixels = pixels[mask]
    
    # Nếu vẫn không có pixel nào
    if len(filtered_pixels) == 0:
        return (0, 0, 0)
    
    # Áp dụng K-means để tìm 3 màu chủ đạo
    n_clusters = min(3, len(filtered_pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(filtered_pixels, n_clusters, None, 
                                     criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Đếm số lượng pixel cho mỗi cluster
    label_counts = np.bincount(labels.flatten())
    
    # Lấy màu của cluster có nhiều pixel nhất
    dominant_idx = np.argmax(label_counts)
    dominant_color = centers[dominant_idx]
    
    return tuple(map(int, dominant_color))

def detect_wire_colors(frame, metal_pins):
    # Vùng dây cắm (phía trái của connector)
    wire_region_width = 30  # 30px từ trái connector
    
    wire_colors = []
    
    for idx, (pin_x, pin_y, pin_w, pin_h) in enumerate(metal_pins):
        # Lấy vùng dây tương ứng với chấu kim loại này
        # Vùng lấy mẫu: từ trái connector đến vị trí chấu, cùng độ cao với chấu
        sample_x = pin_x - 90
        sample_y = pin_y
        sample_w = pin_w
        sample_h = pin_h
        
        # Đảm bảo vùng lấy mẫu hợp lệ
        if sample_w < 5 or sample_h < 5:
            sample_x = max(0, pin_x - 20)
            sample_w = 15
        
        # Đảm bảo không vượt quá biên ảnh
        sample_x = max(0, sample_x)
        sample_y = max(0, sample_y)
        sample_w = min(sample_w, frame.shape[1] - sample_x)
        sample_h = min(sample_h, frame.shape[0] - sample_y)
        
        # Lấy vùng ROI (Region of Interest)
        roi = frame[sample_y:sample_y+sample_h, sample_x:sample_x+sample_w]
        
        if roi.size == 0:
            wire_colors.append({
                'index': idx + 1,
                'color': 'Unknown',
                'bgr': (0, 0, 0),
                'position': (pin_x, pin_y),
                'sample_region': (sample_x, sample_y, sample_w, sample_h)
            })
            continue
        
        # Lấy màu chủ đạo thay vì màu trung bình
        dominant_color = get_dominant_color(roi)
        
        # Xác định tên màu
        color_name = get_color_name(dominant_color)
        
        wire_colors.append({
            'index': idx + 1,
            'color': color_name,
            'bgr': dominant_color,
            'position': (pin_x, pin_y),
            'sample_region': (sample_x, sample_y, sample_w, sample_h)
        })
    
    return wire_colors

def main():
    # Đọc ảnh từ file
    image_path = "4.jpg"  # Thay đổi đường dẫn ảnh của bạn
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Không thể đọc ảnh từ: {image_path}")
        print("Vui lòng kiểm tra đường dẫn file!")
        return
    
    metal_pins = detect_metal_connectors(frame)
    
    # Phát hiện màu sắc các dây
    wire_colors = detect_wire_colors(frame, metal_pins)
    
    # Tạo bản sao để vẽ
    result_img = frame.copy()
    
    # Vẽ các chấu kim loại và vùng lấy mẫu màu
    for i, wire_info in enumerate(wire_colors):
        pin_x, pin_y = wire_info['position']
        sample_x, sample_y, sample_w, sample_h = wire_info['sample_region']
        
        # Vẽ vùng lấy mẫu (màu cyan)
        cv2.rectangle(result_img, (sample_x, sample_y), 
                     (sample_x + sample_w, sample_y + sample_h), 
                     (255, 255, 0), 1)
        
        # Vẽ nhãn màu
        label = f"{i+1}. {wire_info['color']}"
        label_y = sample_y - 5 if sample_y > 20 else sample_y + sample_h + 15
        
        # # Vẽ background cho text
        # (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # cv2.rectangle(result_img, (sample_x, label_y - text_h - 2),
        #              (sample_x + text_w, label_y + 2), (0, 0, 0), -1)
        
        # # Vẽ text
        # cv2.putText(result_img, label, (sample_x, label_y), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Vẽ điểm màu (mẫu màu thực tế)
        color_sample_x = sample_x + sample_w + 5
        color_sample_y = sample_y + sample_h // 2
        cv2.circle(result_img, (color_sample_x, color_sample_y), 5, 
                  wire_info['bgr'], -1)
        cv2.circle(result_img, (color_sample_x, color_sample_y), 5, 
                  (0, 0, 0), 1)
    
    # In thông tin ra console
    print(f"\n{'='*50}")
    print(f"Kết quả phát hiện:")
    print(f"{'='*50}")
    print(f"Tổng số chấu kim loại: {len(wire_colors)}")
    print(f"Thứ tự màu từ TRÊN xuống DƯỚI:")
    print(f"{'-'*50}")
    
    for wire_info in wire_colors:
        bgr = wire_info['bgr']
        print(f"  {wire_info['index']}. {wire_info['color']:12s} - BGR: {bgr}")
    
    print(f"{'='*50}\n")
    
    # Lưu ảnh kết quả
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"Đã lưu ảnh kết quả: {output_path}")
    
    # Hiển thị ảnh
    cv2.imshow('Original Image', frame)
    cv2.imshow('Detection Result', result_img)
    
    print("Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()