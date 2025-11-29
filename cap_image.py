import cv2
import numpy as np
import os
from datetime import datetime

# ======================
# ROI SELECTOR
# ======================
class ROISelector:
    """Class để vẽ ROI bằng chuột"""
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.roi = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                param['current_point'] = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi = (
                min(self.start_point[0], x),
                min(self.start_point[1], y),
                abs(x - self.start_point[0]),
                abs(y - self.start_point[1])
            )
            param['roi_set'] = True
            print(f"✅ ROI: {self.roi}")

def crop_roi_region(image, roi):
    """Crop ảnh theo ROI"""
    if roi is None:
        return None
    
    x, y, w, h = roi
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None
    
    return image[y:y+h, x:x+w]

# ======================
# DATA COLLECTION
# ======================
def collect_data(output_folder='data', class_name=None):
    """
    Thu thập ảnh data với ROI
    - Vẽ ROI trên camera
    - Chụp và lưu ảnh vào thư mục class
    """
    # Tạo thư mục output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Nếu không có class_name, hỏi người dùng
    if class_name is None:
        class_name = input("Nhập tên class (vd: type1, type2, ...): ").strip()
    
    class_folder = os.path.join(output_folder, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
        print(f"✅ Đã tạo thư mục: {class_folder}")
    else:
        print(f"✅ Sử dụng thư mục: {class_folder}")
    
    # Đếm số ảnh hiện có
    existing_images = [f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.png'))]
    image_count = len(existing_images)
    
    print("\n" + "="*60)
    print("DATA COLLECTION TOOL")
    print("="*60)
    print(f"📁 Class: {class_name}")
    print(f"📊 Số ảnh hiện có: {image_count}")
    print("="*60)
    print("📍 HƯỚNG DẪN:")
    print("  1. Kéo chuột để vẽ ROI")
    print("  2. Nhấn ENTER để xác nhận ROI")
    print("  3. Nhấn SPACE để chụp ảnh")
    print("  4. Nhấn 'r' để vẽ lại ROI")
    print("  5. Nhấn 'q' để thoát")
    print("="*60 + "\n")
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    # Setup ROI selector
    roi_selector = ROISelector()
    window_name = 'Data Collection - Draw ROI'
    cv2.namedWindow(window_name)
    
    param = {'current_point': None, 'roi_set': False}
    cv2.setMouseCallback(window_name, roi_selector.mouse_callback, param)
    
    roi_confirmed = False
    captured_count = 0
    
    print("🎯 Vẽ ROI trên camera...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Vẽ ROI đang được kéo
        if roi_selector.drawing and roi_selector.start_point and param['current_point']:
            cv2.rectangle(display, roi_selector.start_point, param['current_point'], (0, 255, 0), 2)
        
        # Vẽ ROI đã hoàn thành
        if roi_selector.roi:
            x, y, w, h = roi_selector.roi
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if not roi_confirmed:
                cv2.putText(display, "Press ENTER to confirm ROI", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Press SPACE to capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Captured: {captured_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display, "Draw ROI by dragging mouse", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Thông tin thêm
        cv2.putText(display, f"Class: {class_name}", (10, display.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Total images: {image_count + captured_count}", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # ENTER - Xác nhận ROI
        if key == 13 and roi_selector.roi and not roi_confirmed:
            roi_confirmed = True
            print(f"✅ ROI đã được xác nhận: {roi_selector.roi}")
            print("📸 Nhấn SPACE để chụp ảnh...")
        
        # SPACE - Chụp ảnh
        elif key == 32 and roi_confirmed:  # SPACE
            # Crop ROI
            cropped = crop_roi_region(frame, roi_selector.roi)
            
            if cropped is not None:
                # Tạo tên file với timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{class_name}_{image_count + captured_count + 1:04d}_{timestamp}.jpg"
                filepath = os.path.join(class_folder, filename)
                
                # Lưu ảnh đã crop
                cv2.imwrite(filepath, cropped)
                captured_count += 1
                print(f"✅ Đã lưu: {filename}")
                
                # Hiệu ứng chụp (flash)
                flash = np.ones_like(display) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(100)
            else:
                print("❌ Không thể crop ROI!")
        
        # r - Vẽ lại ROI
        elif key == ord('r'):
            roi_selector.roi = None
            roi_confirmed = False
            param['current_point'] = None
            param['roi_set'] = False
            print("🔄 Vẽ lại ROI...")
        
        # q - Thoát
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("📊 KẾT QUẢ THU THẬP:")
    print("="*60)
    print(f"✅ Đã chụp: {captured_count} ảnh")
    print(f"📁 Tổng số ảnh trong {class_name}: {image_count + captured_count}")
    print(f"📂 Lưu tại: {class_folder}")
    print("="*60 + "\n")

# ======================
# MAIN
# ======================
def main():
    print("\n" + "="*60)
    print("DATA COLLECTION TOOL - THU THẬP DỮ LIỆU")
    print("="*60)
    
    # Cấu hình
    output_folder = input("Thư mục lưu data (Enter = 'data'): ").strip()
    if not output_folder:
        output_folder = 'data'
    
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Thu thập data cho class mới")
        print("2. Thu thập thêm data cho class đã có")
        print("3. Xem danh sách classes")
        print("4. Thoát")
        print("="*60)
        
        choice = input("Nhập lựa chọn (1/2/3/4): ").strip()
        
        if choice == "1":
            # Thu thập cho class mới
            class_name = input("Nhập tên class mới: ").strip()
            if class_name:
                collect_data(output_folder, class_name)
            else:
                print("⚠️ Tên class không hợp lệ!")
        
        elif choice == "2":
            # Thu thập thêm cho class đã có
            if os.path.exists(output_folder):
                existing_classes = [d for d in os.listdir(output_folder) 
                                   if os.path.isdir(os.path.join(output_folder, d))]
                
                if existing_classes:
                    print("\nCác class đã có:")
                    for i, cls in enumerate(existing_classes, 1):
                        class_folder = os.path.join(output_folder, cls)
                        img_count = len([f for f in os.listdir(class_folder) 
                                        if f.endswith(('.jpg', '.png'))])
                        print(f"  {i}. {cls} ({img_count} ảnh)")
                    
                    class_name = input("\nNhập tên class: ").strip()
                    if class_name in existing_classes:
                        collect_data(output_folder, class_name)
                    else:
                        print("⚠️ Class không tồn tại!")
                else:
                    print("⚠️ Chưa có class nào!")
            else:
                print("⚠️ Thư mục data chưa tồn tại!")
        
        elif choice == "3":
            # Xem danh sách classes
            if os.path.exists(output_folder):
                existing_classes = [d for d in os.listdir(output_folder) 
                                   if os.path.isdir(os.path.join(output_folder, d))]
                
                if existing_classes:
                    print("\n" + "="*60)
                    print("DANH SÁCH CLASSES")
                    print("="*60)
                    total_images = 0
                    for i, cls in enumerate(existing_classes, 1):
                        class_folder = os.path.join(output_folder, cls)
                        img_count = len([f for f in os.listdir(class_folder) 
                                        if f.endswith(('.jpg', '.png'))])
                        total_images += img_count
                        print(f"{i}. {cls}: {img_count} ảnh")
                    print("="*60)
                    print(f"Tổng: {len(existing_classes)} classes, {total_images} ảnh")
                    print("="*60)
                else:
                    print("⚠️ Chưa có class nào!")
            else:
                print("⚠️ Thư mục data chưa tồn tại!")
        
        elif choice == "4":
            print("\n👋 Tạm biệt!")
            break
        
        else:
            print("⚠️ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()