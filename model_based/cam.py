import cv2
import numpy as np
import threading
import time

class WireDetector:
    def __init__(self):
        self.frame = None
        self.result_frame = None
        self.wire_colors = []
        self.is_running = False
        self.lock = threading.Lock()
        self.fps = 0
        self.last_detection_time = 0
        self.exposure = -6  # Giá trị exposure mặc định
        self.cap = None
        
    def on_exposure_change(self, value):
        """Callback khi thay đổi exposure"""
        self.exposure = value - 13  # Chuyển từ 0-13 sang -13-0
        if self.cap is not None:
            # Tắt auto exposure
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
            # Set exposure value
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            print(f"Exposure changed to: {self.exposure}")
    
    def detect_metal_connectors(self, frame):
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
    
    def get_wire_region(self, frame, metal_pins):
        """Xác định vùng chứa các dây"""
        if not metal_pins:
            return None
        
        leftmost_x = min([x for x, y, w, h in metal_pins])
        wire_region_x = 0
        wire_region_width = leftmost_x - 10
        
        top_y = min([y for x, y, w, h in metal_pins]) - 20
        bottom_y = max([y + h for x, y, w, h in metal_pins]) + 20
        
        top_y = max(0, top_y)
        bottom_y = min(frame.shape[0], bottom_y)
        wire_region_width = max(10, wire_region_width)
        
        return (wire_region_x, top_y, wire_region_width, bottom_y - top_y)
    
    def detect_individual_wires(self, frame, wire_region):
        """Detect từng dây riêng lẻ"""
        x, y, w, h = wire_region
        roi = frame[y:y+h, x:x+w]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_color = np.array([0, 30, 30])
        upper_color = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)
        
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wires = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 5000:
                wx, wy, ww, wh = cv2.boundingRect(contour)
                
                if ww > 10 and wh > 5:
                    global_x = x + wx
                    global_y = y + wy
                    
                    wires.append({
                        'bbox': (global_x, global_y, ww, wh),
                        'contour': contour,
                        'area': area,
                        'center_y': global_y + wh // 2
                    })
        
        wires.sort(key=lambda w: w['center_y'])
        return wires
    
    def get_color_name(self, bgr_color):
        """Xác định tên màu"""
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
    
    def get_dominant_color(self, roi):
        """Lấy màu chủ đạo bằng K-means"""
        pixels = roi.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 220)
        
        if np.sum(mask) < 10:
            filtered_pixels = pixels
        else:
            filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            return (0, 0, 0)
        
        n_clusters = min(3, len(filtered_pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(filtered_pixels, n_clusters, None, 
                                         criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        label_counts = np.bincount(labels.flatten())
        dominant_idx = np.argmax(label_counts)
        dominant_color = centers[dominant_idx]
        
        return tuple(map(int, dominant_color))
    
    def detect_wire_colors(self, frame, wires):
        """Xác định màu của từng dây"""
        wire_colors = []
        
        for idx, wire in enumerate(wires):
            wx, wy, ww, wh = wire['bbox']
            
            margin_x = int(ww * 0.15)
            margin_y = int(wh * 0.15)
            
            sample_x = wx + margin_x
            sample_y = wy + margin_y
            sample_w = max(1, ww - 2 * margin_x)
            sample_h = max(1, wh - 2 * margin_y)
            
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
            
            dominant_color = self.get_dominant_color(roi)
            color_name = self.get_color_name(dominant_color)
            
            wire_colors.append({
                'index': idx + 1,
                'color': color_name,
                'bgr': dominant_color,
                'bbox': (wx, wy, ww, wh)
            })
        
        return wire_colors
    
    def detection_thread(self):
        """Thread xử lý detection liên tục"""
        while self.is_running:
            with self.lock:
                if self.frame is None:
                    continue
                frame = self.frame.copy()
            
            start_time = time.time()
            
            # Detect
            metal_pins = self.detect_metal_connectors(frame)
            
            if metal_pins is not None and len(metal_pins) > 0:
                wire_region = self.get_wire_region(frame, metal_pins)
                wires = self.detect_individual_wires(frame, wire_region)
                wire_colors = self.detect_wire_colors(frame, wires)
                
                # Vẽ kết quả
                result = frame.copy()
                
                # Vẽ vùng chứa dây
                if wire_region:
                    x, y, w, h = wire_region
                    cv2.rectangle(result, (x, y), (x + w, y + h), (128, 128, 128), 1)
                
                # Vẽ từng dây
                for wire_info in wire_colors:
                    wx, wy, ww, wh = wire_info['bbox']
                    
                    # Bounding box
                    cv2.rectangle(result, (wx, wy), (wx + ww, wy + wh), (0, 255, 0), 2)
                    
                    # Label
                    label = f"{wire_info['index']}. {wire_info['color']}"
                    label_y = wy - 5 if wy > 20 else wy + wh + 15
                    
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(result, (wx, label_y - text_h - 2),
                                 (wx + text_w, label_y + 2), (0, 0, 0), -1)
                    cv2.putText(result, label, (wx, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Color sample
                    color_sample_x = wx + ww + 10
                    color_sample_y = wy + wh // 2
                    cv2.circle(result, (color_sample_x, color_sample_y), 8, 
                              wire_info['bgr'], -1)
                    cv2.circle(result, (color_sample_x, color_sample_y), 8, 
                              (0, 0, 0), 2)
                
                with self.lock:
                    self.result_frame = result
                    self.wire_colors = wire_colors
            
            # Tính FPS
            time.sleep(0.001)  # Đảm bảo time difference không bằng 0
            self.fps = 1.0 / (time.time() - start_time)
            self.last_detection_time = time.time()
            
            # Giảm tải CPU
            time.sleep(0.05)
    
    def run(self):
        """Chạy camera và detection"""
        # Mở camera
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Không thể mở camera!")
            return
        
        # Set resolution (tùy chọn)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set manual exposure mode
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        
        # Tạo cửa sổ với trackbar
        window_name = 'Wire Color Detector'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 900, 480)
        
        # Tạo trackbar cho exposure (0-13 tương ứng với -13 đến 0)
        cv2.createTrackbar('Exposure', window_name, 7, 13, self.on_exposure_change)
        
        # Bắt đầu detection thread
        self.is_running = True
        detection_thread = threading.Thread(target=self.detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        
        print("=" * 60)
        print("WIRE COLOR DETECTOR - Real-time")
        print("=" * 60)
        print("Nhấn 'q' để thoát")
        print("Nhấn 's' để lưu ảnh kết quả")
        print("Nhấn 'p' để in kết quả ra console")
        print("Sử dụng thanh trượt 'Exposure' để điều chỉnh độ sáng")
        print("=" * 60)
        
        frame_count = 0
        display_fps = 0
        fps_update_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Cập nhật frame
            with self.lock:
                self.frame = frame.copy()
                display_frame = self.result_frame if self.result_frame is not None else frame
                current_wire_colors = self.wire_colors.copy()
            
            # Tính FPS hiển thị
            frame_count += 1
            if time.time() - fps_update_time > 1.0:
                display_fps = frame_count
                frame_count = 0
                fps_update_time = time.time()
            
            # Tạo panel bên phải cho thông tin
            panel_width = 260
            panel_height = display_frame.shape[0]
            panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)  # Màu nền xám đậm
            
            # Vẽ thông tin lên panel
            y_offset = 30
            
            # Tiêu đề
            cv2.putText(panel, "CONTROLS", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            
            # Thông tin FPS
            cv2.putText(panel, f"FPS: {display_fps}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 30
            
            # Thông tin Exposure
            cv2.putText(panel, f"Exposure: {self.exposure}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            y_offset += 40
            
            # Đường phân cách
            cv2.line(panel, (10, y_offset), (panel_width-10, y_offset), (100, 100, 100), 1)
            y_offset += 20
            
            # Thông tin số dây
            if current_wire_colors:
                cv2.putText(panel, "DETECTED WIRES", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
                
                cv2.putText(panel, f"Total: {len(current_wire_colors)}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 35
                
                # Liệt kê màu các dây
                for wire_info in current_wire_colors:
                    if y_offset > panel_height - 30:
                        break
                    
                    # Vẽ số thứ tự và tên màu
                    text = f"{wire_info['index']}. {wire_info['color']}"
                    cv2.putText(panel, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Vẽ color sample
                    color_x = panel_width - 40
                    color_y = y_offset - 8
                    cv2.circle(panel, (color_x, color_y), 12, wire_info['bgr'], -1)
                    cv2.circle(panel, (color_x, color_y), 12, (255, 255, 255), 1)
                    
                    y_offset += 25
            else:
                cv2.putText(panel, "NO WIRES", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 30
                cv2.putText(panel, "DETECTED", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Hướng dẫn phím tắt
            y_offset = panel_height - 100
            cv2.line(panel, (10, y_offset), (panel_width-10, y_offset), (100, 100, 100), 1)
            y_offset += 20
            
            cv2.putText(panel, "SHORTCUTS", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(panel, "Q - Quit", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
            cv2.putText(panel, "S - Save image", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
            cv2.putText(panel, "P - Print result", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Ghép panel vào bên phải display frame
            combined_frame = np.hstack([display_frame, panel])
            
            # Hiển thị
            cv2.imshow(window_name, combined_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"wire_detection_{timestamp}.jpg"
                cv2.imwrite(filename, combined_frame)
                print(f"\nĐã lưu ảnh: {filename}")
            elif key == ord('p'):
                if current_wire_colors:
                    print(f"\n{'='*60}")
                    print(f"KẾT QUẢ PHÁT HIỆN (thời điểm: {time.strftime('%H:%M:%S')})")
                    print(f"{'='*60}")
                    print(f"Tổng số dây: {len(current_wire_colors)}")
                    print(f"{'-'*60}")
                    for wire_info in current_wire_colors:
                        bgr = wire_info['bgr']
                        print(f"  {wire_info['index']}. {wire_info['color']:12s} - BGR: {bgr}")
                    print(f"{'='*60}\n")
                else:
                    print("\nChưa phát hiện được dây nào!")
        
        # Dọn dẹp
        self.is_running = False
        detection_thread.join(timeout=1)
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nĐã dừng chương trình.")

def main():
    detector = WireDetector()
    detector.run()

if __name__ == "__main__":
    main()