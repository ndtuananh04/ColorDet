import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_2d import SiameseMobileNet
import os

# ======================
# ROI SELECTOR
# ======================
class ROISelectorRealtime:
    """Class để di chuyển ROI cố định trong real-time camera"""
    def __init__(self, roi_width=30, roi_height=250):
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.roi_x = 0
        self.roi_y = 0
        self.dragging = False
        self.offset_x = 0
        self.offset_y = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.roi_x <= x <= self.roi_x + self.roi_width and 
                self.roi_y <= y <= self.roi_y + self.roi_height):
                self.dragging = True
                self.offset_x = x - self.roi_x
                self.offset_y = y - self.roi_y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.roi_x = x - self.offset_x
                self.roi_y = y - self.offset_y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
    
    def get_roi(self):
        return (self.roi_x, self.roi_y, self.roi_width, self.roi_height)
    
    def clamp_roi(self, frame_width, frame_height):
        self.roi_x = max(0, min(self.roi_x, frame_width - self.roi_width))
        self.roi_y = max(0, min(self.roi_y, frame_height - self.roi_height))

def input_roi_size():
    """Nhập kích thước ROI"""
    print("\n" + "="*60)
    print("⚙️  CÀI ĐẶT KÍCH THƯỚC ROI")
    print("="*60)
    
    width = input("   Width [30]: ").strip()
    width = int(width) if width else 30
    
    height = input("   Height [250]: ").strip()
    height = int(height) if height else 250
    
    print(f"✅ ROI: {width}x{height}px")
    return width, height

# ======================
# SETUP
# ======================
def load_model(model_path, device='cuda'):
    """Load trained model"""
    model = SiameseMobileNet(embedding_dim=128, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f"✅ Model loaded (Val Loss: {checkpoint.get('val_loss', 0):.4f})")
    return model

def get_transform():
    """Transform cho ảnh"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(img, transform):
    """Preprocess OpenCV image hoặc file path"""
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = Image.fromarray(img)
    return transform(img).unsqueeze(0)

# ======================
# CHỨC NĂNG 1: SO SÁNH 2 ẢNH
# ======================
def compare_two_images(model, img1_path, img2_path, transform, device, threshold=0.5):
    """So sánh 2 ảnh từ file"""
    model.eval()
    
    with torch.no_grad():
        img1 = preprocess_image(img1_path, transform).to(device)
        img2 = preprocess_image(img2_path, transform).to(device)
        distance = model(img1, img2).item()
    
    is_same = distance < threshold
    
    print(f"\n{'='*50}")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print(f"Distance: {distance:.4f}")
    print(f"Result: {'✅ SAME CLASS' if is_same else '❌ DIFFERENT CLASS'}")
    print(f"{'='*50}\n")
    
    return distance, is_same

# ======================
# CHỨC NĂNG 2: CAMERA REAL-TIME WITH ROI
# ======================
def realtime_camera_with_roi(model, reference_img_path, transform, device, threshold=0.5, 
                             roi_width=30, roi_height=250, camera_id=0):
    """
    So sánh real-time từ camera với ảnh mẫu, sử dụng ROI có thể di chuyển
    """
    model.eval()
    
    # Load ảnh mẫu
    print(f"📷 Loading reference: {reference_img_path}")
    with torch.no_grad():
        ref_img = preprocess_image(reference_img_path, transform).to(device)
    
    # Mở camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read camera")
        cap.release()
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    
    # Setup ROI
    roi_selector = ROISelectorRealtime(roi_width=roi_width, roi_height=roi_height)
    roi_selector.roi_x = (frame_width - roi_width) // 2
    roi_selector.roi_y = (frame_height - roi_height) // 2
    
    window_name = 'Real-time Detection with ROI'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, roi_selector.mouse_callback)
    
    print(f"\n{'='*60}")
    print("🎥 REAL-TIME DETECTION WITH ROI")
    print(f"{'='*60}")
    print(f"ROI: {roi_width}x{roi_height}px | Threshold: {threshold}")
    print("Controls: q=Quit | s=Save | +/-=Threshold | Drag=Move ROI")
    print(f"{'='*60}\n")
    
    import time
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        roi_selector.clamp_roi(frame_width, frame_height)
        x, y, w, h = roi_selector.get_roi()
        
        try:
            # Crop ROI
            roi_frame = frame[y:y+h, x:x+w]
            if roi_frame.size == 0:
                raise ValueError("Empty ROI")
            
            # So sánh ROI với reference
            with torch.no_grad():
                current_img = preprocess_image(roi_frame, transform).to(device)
                distance = model(ref_img, current_img).item()
            
            is_match = distance < threshold
            
            # Màu ROI
            if roi_selector.dragging:
                roi_color = (0, 255, 255)  # Vàng
            elif is_match:
                roi_color = (0, 255, 0)    # Xanh
            else:
                roi_color = (0, 0, 255)    # Đỏ
            
            cv2.rectangle(display, (x, y), (x+w, y+h), roi_color, 3)
            
            # Tâm ROI
            cv2.circle(display, (x + w//2, y + h//2), 5, (255, 0, 255), -1)
            
            # Hiển thị info
            status = "MATCH" if is_match else "DIFFERENT"
            status_color = (0, 255, 0) if is_match else (0, 0, 255)
            
            cv2.putText(display, status, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
            cv2.putText(display, f"Distance: {distance:.4f}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Threshold: {threshold:.4f}", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"ROI: {w}x{h}px", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
            
            # FPS
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # ROI Preview
            preview_size = 150
            roi_resized = cv2.resize(roi_frame, (preview_size, int(preview_size * h / w)))
            preview_y = display.shape[0] - roi_resized.shape[0] - 10
            preview_x = display.shape[1] - preview_size - 10
            display[preview_y:preview_y+roi_resized.shape[0], 
                   preview_x:preview_x+preview_size] = roi_resized
            cv2.rectangle(display, (preview_x, preview_y), 
                         (preview_x+preview_size, preview_y+roi_resized.shape[0]), 
                         (255, 255, 255), 2)
            
        except Exception as e:
            cv2.putText(display, f"Error: {str(e)[:50]}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.putText(display, "Drag: Move | q: Quit | s: Save | +/-: Threshold", 
                   (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'roi_captured_{int(time.time())}.jpg'
            cv2.imwrite(filename, roi_frame)
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
# CHỨC NĂNG 3: CAMERA REAL-TIME (FULL FRAME - CŨ)
# ======================
def realtime_camera_compare(model, reference_img_path, transform, device, threshold=0.5, camera_id=0):
    """So sánh real-time từ camera với ảnh mẫu (full frame)"""
    model.eval()
    
    # Load ảnh mẫu
    print(f"📷 Loading reference image: {reference_img_path}")
    with torch.no_grad():
        ref_img = preprocess_image(reference_img_path, transform).to(device)
    
    # Mở camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print(f"\n{'='*60}")
    print("🎥 REAL-TIME CAMERA DETECTION")
    print(f"{'='*60}")
    print(f"Reference image: {reference_img_path}")
    print(f"Threshold: {threshold}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save frame")
    print("  '+' - Increase threshold (+0.1)")
    print("  '-' - Decrease threshold (-0.1)")
    print(f"{'='*60}\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break
        
        # Resize để hiển thị
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # So sánh với ảnh mẫu
        with torch.no_grad():
            current_img = preprocess_image(frame, transform).to(device)
            distance = model(ref_img, current_img).item()
        
        is_same = distance < threshold
        
        # Vẽ kết quả lên frame
        if is_same:
            color = (0, 255, 0)  # Green
            status = "MATCH"
        else:
            color = (0, 0, 255)  # Red
            status = "NO MATCH"
        
        # Vẽ border
        cv2.rectangle(display_frame, (10, 10), (w-10, h-10), color, 5)
        
        # Hiển thị thông tin
        cv2.putText(display_frame, f"Status: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(display_frame, f"Distance: {distance:.3f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Threshold: {threshold:.2f}", (20, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Hiển thị controls
        cv2.putText(display_frame, "Q: Quit | S: Save | +/-: Threshold", 
                    (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Real-time Detection', display_frame)
        
        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Exiting...")
            break
        elif key == ord('s'):
            filename = f"captured_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved: {filename} (distance={distance:.4f})")
            frame_count += 1
        elif key == ord('+') or key == ord('='):
            threshold += 0.1
            print(f"📈 Threshold increased: {threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            threshold = max(0.1, threshold - 0.1)
            print(f"📉 Threshold decreased: {threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera closed")

# ======================
# MAIN
# ======================
def main():
    # Cấu hình
    model_path = "siamese_mobilenet_model.pth"
    device = 'cpu'
    threshold = 0.5
    
    # Load model
    model = load_model(model_path, device)
    transform = get_transform()
    
    print("\n" + "="*60)
    print("SIAMESE MODEL - INFERENCE")
    print("="*60)
    print("\nChọn chức năng:")
    print("  1. So sánh 2 ảnh từ file")
    print("  2. Camera real-time với ảnh mẫu (full frame)")
    print("  3. Camera real-time với ROI (30x250)")
    print("="*60)
    
    choice = input("\nNhập lựa chọn (1/2/3): ").strip()
    
    if choice == '1':
        # So sánh 2 ảnh
        img1 = input("Đường dẫn ảnh 1: ").strip()
        img2 = input("Đường dẫn ảnh 2: ").strip()
        
        if os.path.exists(img1) and os.path.exists(img2):
            compare_two_images(model, img1, img2, transform, device, threshold)
        else:
            print("❌ Ảnh không tồn tại!")
    
    elif choice == '2':
        # Camera real-time (full frame)
        ref_img = input("Đường dẫn ảnh mẫu: ").strip()
        camera_id = input("Camera ID (mặc định 0): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        
        if os.path.exists(ref_img):
            realtime_camera_compare(model, ref_img, transform, device, threshold, camera_id)
        else:
            print("❌ Ảnh mẫu không tồn tại!")
    
    elif choice == '3':
        # Camera real-time với ROI
        ref_img = input("Đường dẫn ảnh mẫu: ").strip()
        
        if not os.path.exists(ref_img):
            print("❌ Ảnh mẫu không tồn tại!")
            return
        
        # Nhập kích thước ROI
        roi_width, roi_height = input_roi_size()
        
        camera_id = input("Camera ID (mặc định 0): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        
        realtime_camera_with_roi(model, ref_img, transform, device, threshold, 
                                roi_width, roi_height, camera_id)
    
    else:
        print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()