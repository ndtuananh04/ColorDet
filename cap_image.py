import cv2
import numpy as np
import os
from datetime import datetime

# ======================
# ROI SELECTOR
# ======================
class ROISelector:
    """Class để di chuyển ROI cố định với kích thước tùy chỉnh"""
    def __init__(self, roi_width=30, roi_height=180):
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
# INPUT ROI SIZE
# ======================
def input_roi_size():
    """
    Nhập kích thước ROI từ người dùng
    Returns: (width, height) tuple
    """
    print("\n" + "="*60)
    print("⚙️  CÀI ĐẶT KÍCH THƯỚC ROI")
    print("="*60)
    
    while True:
        try:
            print("\n📏 Nhập kích thước ROI (pixels):")
            print("   Gợi ý: 30x180 (dọc), 250x30 (ngang)")
            
            width_input = input("   Width (chiều rộng): ").strip()
            if not width_input:
                width = 30  # Default
                print(f"   → Sử dụng giá trị mặc định: {width}px")
            else:
                width = int(width_input)
            
            height_input = input("   Height (chiều cao): ").strip()
            if not height_input:
                height = 180  # Default
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
# DATA COLLECTION
# ======================
def collect_data(output_folder='data', class_name=None, roi_width=30, roi_height=180):
    """
    Thu thập ảnh data với ROI cố định
    - Di chuyển ROI bằng cách kéo thả
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
    print(f"📏 ROI size: {roi_width}x{roi_height} pixels (cố định)")
    print("="*60)
    print("📍 HƯỚNG DẪN:")
    print("  1. Kéo thả ROI (hình chữ nhật xanh) để di chuyển")
    print("  2. Nhấn SPACE để chụp ảnh")
    print("  3. Nhấn 'e' để điều chỉnh exposure")
    print("  4. Nhấn 'q' để thoát")
    print("="*60 + "\n")
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    # Đọc frame đầu tiên để lấy kích thước
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame từ camera!")
        cap.release()
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    
    # Kiểm tra ROI có vừa với frame không
    if roi_width > frame_width or roi_height > frame_height:
        print(f"❌ ROI ({roi_width}x{roi_height}) lớn hơn frame ({frame_width}x{frame_height})!")
        cap.release()
        return
    
    # ============ HỎI ĐIỀU CHỈNH EXPOSURE ============
    print("\n📸 Bạn có muốn điều chỉnh exposure không?")
    adjust_choice = input("   (y/n) [n]: ").strip().lower()
    if adjust_choice == 'y':
        adjust_camera_exposure(cap)
    # ================================================
    
    # Setup ROI selector với kích thước tùy chỉnh
    roi_selector = ROISelector(roi_width=roi_width, roi_height=roi_height)
    
    # Đặt ROI ở giữa màn hình
    roi_selector.roi_x = (frame_width - roi_width) // 2
    roi_selector.roi_y = (frame_height - roi_height) // 2
    
    window_name = 'Data Collection - Move ROI'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, roi_selector.mouse_callback)
    
    captured_count = 0
    
    print(f"🎯 ROI cố định: {roi_width}x{roi_height} pixels")
    print("📸 Kéo thả ROI để điều chỉnh vị trí, sau đó nhấn SPACE để chụp...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Giới hạn ROI trong frame
        roi_selector.clamp_roi(frame_width, frame_height)
        
        # Lấy vị trí ROI hiện tại
        x, y, w, h = roi_selector.get_roi()
        
        # Vẽ ROI
        if roi_selector.dragging:
            # Màu vàng khi đang kéo
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
        else:
            # Màu xanh lá khi không kéo
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Vẽ tâm ROI
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(display, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Hiển thị kích thước ROI
        cv2.putText(display, f"ROI: {w}x{h}px", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hiển thị hướng dẫn
        cv2.putText(display, "Drag: Move | SPACE: Capture | E: Exposure | Q: Quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Captured: {captured_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Thông tin thêm
        cv2.putText(display, f"Class: {class_name}", (10, display.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Total images: {image_count + captured_count}", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # SPACE - Chụp ảnh
        if key == 32:  # SPACE
            # Crop ROI
            cropped = crop_roi_region(frame, roi_selector.get_roi())
            
            if cropped is not None:
                # Tạo tên file với timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{class_name}_{image_count + captured_count + 1:04d}_{timestamp}.jpg"
                filepath = os.path.join(class_folder, filename)
                
                # Lưu ảnh đã crop
                cv2.imwrite(filepath, cropped)
                captured_count += 1
                print(f"✅ Đã lưu: {filename} - Shape: {cropped.shape}")
                
                # Hiệu ứng chụp (flash)
                flash = np.ones_like(display) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(100)
            else:
                print("❌ Không thể crop ROI!")
        
        # e - Điều chỉnh exposure
        elif key == ord('e'):
            print("\n📸 Điều chỉnh exposure...")
            adjust_camera_exposure(cap)
        
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
    print(f"📏 Kích thước ảnh: {roi_width}x{roi_height} pixels")
    print("="*60 + "\n")

def adjust_camera_exposure(cap):
    """Điều chỉnh exposure của camera"""
    print("\n⚙️  ĐIỀU CHỈNH EXPOSURE")
    
    # Set manual mode
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    
    # Lấy exposure hiện tại
    current_exp = int(cap.get(cv2.CAP_PROP_EXPOSURE))
    
    # Thử set range và đọc lại
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    test_exp = int(cap.get(cv2.CAP_PROP_EXPOSURE))
    
    # Xác định range dựa trên camera
    if test_exp < 0:
        min_exp, max_exp = -13, -1  # Camera hỗ trợ âm
    else:
        min_exp, max_exp = 1, 2000  # Camera dùng giá trị dương
        if current_exp == 0:
            current_exp = 100
    
    cap.set(cv2.CAP_PROP_EXPOSURE, current_exp)
    
    window = 'Adjust Exposure'
    cv2.namedWindow(window)
    
    def on_change(val):
        # Chuyển đổi từ 0-100 sang min_exp-max_exp
        exp = int(min_exp + (val / 100.0) * (max_exp - min_exp))
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    
    # Trackbar từ 0-100%
    initial_val = int(((current_exp - min_exp) / (max_exp - min_exp)) * 100)
    cv2.createTrackbar('Exposure (%)', window, initial_val, 100, on_change)
    
    print("📍 Di chuyển trackbar | ENTER: Xác nhận | ESC: Hủy")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        exp = int(cap.get(cv2.CAP_PROP_EXPOSURE))
        display = frame.copy()
        
        cv2.putText(display, f"Exposure: {exp}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(display, f"Range: {min_exp} to {max_exp}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "ENTER: Confirm | ESC: Cancel", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            print(f"✅ Exposure: {exp}")
            break
        elif key == 27:  # ESC
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            print("⚠️ Hủy - Auto exposure")
            break
    
    cv2.destroyWindow(window)

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
    
    # Nhập kích thước ROI
    roi_width, roi_height = input_roi_size()
    if roi_width is None or roi_height is None:
        print("❌ Đã hủy!")
        return
    
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print(f"📏 ROI hiện tại: {roi_width}x{roi_height} pixels")
        print("="*60)
        print("1. Thu thập data cho class mới")
        print("2. Thu thập thêm data cho class đã có")
        print("3. Xem danh sách classes")
        print("4. Thay đổi kích thước ROI")
        print("5. Thoát")
        print("="*60)
        
        choice = input("Nhập lựa chọn (1/2/3/4/5): ").strip()
        
        if choice == "1":
            # Thu thập cho class mới
            class_name = input("Nhập tên class mới: ").strip()
            if class_name:
                collect_data(output_folder, class_name, roi_width, roi_height)
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
                        collect_data(output_folder, class_name, roi_width, roi_height)
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
            # Thay đổi kích thước ROI
            new_width, new_height = input_roi_size()
            if new_width is not None and new_height is not None:
                roi_width, roi_height = new_width, new_height
                print(f"✅ Đã cập nhật ROI: {roi_width}x{roi_height} pixels")
        
        elif choice == "5":
            print("\n👋 Tạm biệt!")
            break
        
        else:
            print("⚠️ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()