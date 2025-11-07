import cv2
import numpy as np

def detect_metal_connectors(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Tìm contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc và vẽ bounding box
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Lọc theo diện tích (chỉ giữ lại các vùng có kích thước phù hợp)
        if area > 100 and area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Lọc theo tỷ lệ khung hình
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Đầu nối thường có tỷ lệ gần vuông hoặc hình chữ nhật nhỏ
            if 0.3 < aspect_ratio < 3.0:
                detections.append((x, y, w, h))
                
    return detections

def main():
    # Khởi động camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    
    print("Nhấn 'q' để thoát")
    print("Nhấn 's' để lưu ảnh")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Không thể đọc frame từ camera!")
            break
        
        # Phát hiện các đầu nối kim loại
        detections = detect_metal_connectors(frame)
        
        # Vẽ bounding box lên frame
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Connector", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hiển thị số lượng phát hiện
        cv2.putText(frame, f"Detected: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị frame
        cv2.imshow('Metal Connector Detection', frame)
        
        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('detection_result.jpg', frame)
            print("Đã lưu ảnh!")
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()