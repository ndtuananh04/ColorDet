import cv2
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# ROI INTERACTIVE SELECTION
# ======================
class ROISelector:
    def __init__(self):
        self.roi = None
        self.dragging = False
        self.start_point = None
        self.current_image = None
        self.display_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.display_image = self.current_image.copy()
                cv2.rectangle(self.display_image, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Select ROI', self.display_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            x1, y1 = self.start_point
            x2, y2 = x, y
            
            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.roi = (x1, y1, x2 - x1, y2 - y1)
                self.display_image = self.current_image.copy()
                cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.display_image, "Press ENTER to confirm, 'r' to reset", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Select ROI', self.display_image)
    
    def select_roi(self, image):
        """
        Interactive ROI selection
        Returns: (x, y, w, h) or None
        """
        self.current_image = image.copy()
        self.display_image = image.copy()
        self.roi = None
        
        cv2.namedWindow('Select ROI')
        cv2.setMouseCallback('Select ROI', self.mouse_callback)
        
        print("\n" + "="*60)
        print("CHỌN VÙNG ROI:")
        print("="*60)
        print("- Kéo chuột để chọn vùng dây")
        print("- Nhấn ENTER để xác nhận")
        print("- Nhấn 'r' để chọn lại")
        print("- Nhấn ESC để thoát")
        print("="*60)
        
        cv2.imshow('Select ROI', self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                if self.roi is not None:
                    cv2.destroyWindow('Select ROI')
                    return self.roi
                    
            elif key == ord('r'):  # Reset
                self.roi = None
                self.display_image = self.current_image.copy()
                cv2.imshow('Select ROI', self.display_image)
                
            elif key == 27:  # ESC
                cv2.destroyWindow('Select ROI')
                return None

class ROIManager:
    """Quản lý ROI với khả năng di chuyển và resize"""
    def __init__(self, image, initial_roi=None):
        self.image = image.copy()
        self.display_image = image.copy()
        self.roi = initial_roi  # (x, y, w, h)
        self.dragging = False
        self.resizing = False
        self.drag_start = None
        self.resize_corner = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if self.roi is None:
            return
            
        rx, ry, rw, rh = self.roi
        
        # Check if near corners for resizing
        corner_size = 10
        corners = {
            'tl': (rx, ry),
            'tr': (rx + rw, ry),
            'bl': (rx, ry + rh),
            'br': (rx + rw, ry + rh)
        }
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check resize corners first
            for corner_name, (cx, cy) in corners.items():
                if abs(x - cx) < corner_size and abs(y - cy) < corner_size:
                    self.resizing = True
                    self.resize_corner = corner_name
                    self.drag_start = (x, y)
                    return
            
            # Check if inside ROI for dragging
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                self.dragging = True
                self.drag_start = (x - rx, y - ry)
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.resizing and self.drag_start is not None:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                
                if self.resize_corner == 'br':
                    new_w = rw + dx
                    new_h = rh + dy
                    if new_w > 20 and new_h > 20:
                        self.roi = (rx, ry, new_w, new_h)
                        self.drag_start = (x, y)
                        
                elif self.resize_corner == 'tr':
                    new_w = rw + dx
                    new_y = ry + dy
                    new_h = rh - dy
                    if new_w > 20 and new_h > 20:
                        self.roi = (rx, new_y, new_w, new_h)
                        self.drag_start = (x, y)
                        
                elif self.resize_corner == 'bl':
                    new_x = rx + dx
                    new_w = rw - dx
                    new_h = rh + dy
                    if new_w > 20 and new_h > 20:
                        self.roi = (new_x, ry, new_w, new_h)
                        self.drag_start = (x, y)
                        
                elif self.resize_corner == 'tl':
                    new_x = rx + dx
                    new_y = ry + dy
                    new_w = rw - dx
                    new_h = rh - dy
                    if new_w > 20 and new_h > 20:
                        self.roi = (new_x, new_y, new_w, new_h)
                        self.drag_start = (x, y)
                
                self.update_display()
                
            elif self.dragging and self.drag_start is not None:
                new_x = x - self.drag_start[0]
                new_y = y - self.drag_start[1]
                
                # Keep within image bounds
                new_x = max(0, min(new_x, self.image.shape[1] - rw))
                new_y = max(0, min(new_y, self.image.shape[0] - rh))
                
                self.roi = (new_x, new_y, rw, rh)
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False
            self.drag_start = None
            self.resize_corner = None
    
    def update_display(self):
        if self.roi is None:
            return
            
        self.display_image = self.image.copy()
        x, y, w, h = self.roi
        
        # Draw ROI rectangle
        cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw resize corners
        corner_size = 8
        corners = [
            (x, y), (x + w, y), 
            (x, y + h), (x + w, y + h)
        ]
        for cx, cy in corners:
            cv2.circle(self.display_image, (cx, cy), corner_size, (0, 0, 255), -1)
        
        # Draw info
        cv2.putText(self.display_image, f"ROI: {w}x{h}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Adjust ROI', self.display_image)
    
    def adjust_roi(self):
        """
        Adjust ROI with mouse
        Returns: (x, y, w, h) or None
        """
        cv2.namedWindow('Adjust ROI')
        cv2.setMouseCallback('Adjust ROI', self.mouse_callback)
        
        print("\n" + "="*60)
        print("ĐIỀU CHỈNH ROI:")
        print("="*60)
        print("- Kéo vùng ROI để di chuyển")
        print("- Kéo góc (đỏ) để thay đổi kích thước")
        print("- Nhấn ENTER để xác nhận")
        print("- Nhấn 'd' để xóa ROI và chọn lại")
        print("- Nhấn ESC để thoát")
        print("="*60)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                if self.roi is not None:
                    cv2.destroyWindow('Adjust ROI')
                    return self.roi
                    
            elif key == ord('d'):  # Delete and reselect
                cv2.destroyWindow('Adjust ROI')
                selector = ROISelector()
                return selector.select_roi(self.image)
                
            elif key == 27:  # ESC
                cv2.destroyWindow('Adjust ROI')
                return None

def select_or_input_roi(image, use_interactive=True):
    """
    Chọn ROI bằng cách interactive hoặc nhập tay
    
    Args:
        image: ảnh input
        use_interactive: True = dùng chuột, False = nhập tay
    
    Returns:
        (x, y, w, h) hoặc None
    """
    if use_interactive:
        selector = ROISelector()
        return selector.select_roi(image)
    else:
        print("\nNhập kích thước ROI:")
        try:
            x = int(input("  x (left): "))
            y = int(input("  y (top): "))
            w = int(input("  width: "))
            h = int(input("  height: "))
            
            # Validate
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                print("❌ Giá trị không hợp lệ!")
                return None
            if x + w > image.shape[1] or y + h > image.shape[0]:
                print("❌ ROI vượt quá kích thước ảnh!")
                return None
                
            return (x, y, w, h)
        except ValueError:
            print("❌ Giá trị không hợp lệ!")
            return None

def crop_wire_region(image, roi=None, interactive=True):
    """
    Crop vùng dây theo ROI
    
    Args:
        image: ảnh input
        roi: (x, y, w, h) hoặc None để chọn mới
        interactive: True = dùng chuột để chọn/điều chỉnh
    
    Returns:
        cropped image hoặc None
    """
    if roi is None:
        roi = select_or_input_roi(image, use_interactive=interactive)
        if roi is None:
            return None
    else:
        # Allow adjustment if interactive
        if interactive:
            manager = ROIManager(image, roi)
            roi = manager.adjust_roi()
            if roi is None:
                return None
    
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]
    return cropped

# ======================
# TRÍCH XUẤT ĐẶC TRƯNG MÀU
# ======================
def extract_color_features(img, method='hsv_histogram'):
    """
    Trích xuất đặc trưng màu từ ảnh
    
    Methods:
    - 'hsv_histogram': Histogram HSV (chi tiết nhất)
    - 'mean_hsv': Giá trị trung bình HSV mỗi dòng
    - 'color_moments': Color moments (mean, std, skewness)
    """
    if method == 'hsv_histogram':
        # Chuyển sang HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Tính histogram cho mỗi kênh
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Concatenate
        feature = np.concatenate([h_hist, s_hist, v_hist])
        
    elif method == 'mean_hsv':
        # Resize về chuẩn
        resized = cv2.resize(img, (30, 40))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Lấy giá trị trung bình theo chiều ngang (mỗi dòng dây)
        feature = np.mean(hsv, axis=1).flatten()
        
    elif method == 'color_moments':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Tính mean, std, skewness cho mỗi kênh
        features = []
        for channel in range(3):
            channel_data = hsv[:, :, channel].flatten()
            
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            skewness = np.mean(((channel_data - mean) / (std + 1e-7)) ** 3)
            
            features.extend([mean, std, skewness])
        
        feature = np.array(features)
    
    return feature

# ======================
# SO SÁNH ĐẶC TRƯNG
# ======================
def compare_features(feat1, feat2, method='cosine'):
    """
    So sánh 2 đặc trưng
    
    Methods:
    - 'cosine': Cosine similarity (0-1, càng cao càng giống)
    - 'euclidean': Euclidean distance (càng thấp càng giống)
    - 'correlation': Correlation coefficient
    """
    if method == 'cosine':
        # Cosine similarity (0-1, 1 = giống hệt)
        similarity = 1 - cosine(feat1, feat2)
        return similarity
    
    elif method == 'euclidean':
        # Euclidean distance (0-inf, 0 = giống hệt)
        distance = euclidean(feat1, feat2)
        # Normalize về 0-1 (1 = giống)
        max_dist = np.sqrt(len(feat1))  # Max possible distance
        similarity = 1 - (distance / max_dist)
        return similarity
    
    elif method == 'correlation':
        # Correlation coefficient (-1 to 1, 1 = perfect positive correlation)
        correlation = np.corrcoef(feat1, feat2)[0, 1]
        return correlation

# ======================
# PIPELINE SO SÁNH
# ======================
def compare_two_images_simple(img1_path, img2_path, 
                               feature_method='hsv_histogram',
                               compare_method='cosine',
                               threshold=0.85,
                               roi1=None, roi2=None,
                               interactive=True):
    """
    So sánh 2 ảnh connector đơn giản, không cần ML
    
    Args:
        feature_method: 'hsv_histogram', 'mean_hsv', 'color_moments'
        compare_method: 'cosine', 'euclidean', 'correlation'
        threshold: ngưỡng quyết định (similarity > threshold → giống)
        roi1, roi2: (x, y, w, h) hoặc None để chọn mới
        interactive: True = dùng chuột, False = nhập tay
    """
    # Đọc ảnh
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("❌ Không thể đọc ảnh!")
        return None
    
    # Crop vùng dây
    print("\n--- ẢNH 1 ---")
    crop1 = crop_wire_region(img1, roi1, interactive)
    
    if crop1 is None:
        print("❌ Không có ROI cho ảnh 1")
        return None
    
    print("\n--- ẢNH 2 ---")
    crop2 = crop_wire_region(img2, roi2, interactive)
    
    if crop2 is None:
        print("❌ Không có ROI cho ảnh 2")
        return None
    
    # Trích xuất đặc trưng
    feat1 = extract_color_features(crop1, method=feature_method)
    feat2 = extract_color_features(crop2, method=feature_method)
    
    # So sánh
    similarity = compare_features(feat1, feat2, method=compare_method)
    
    # Kết luận
    is_match = similarity > threshold
    
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ SO SÁNH (KHÔNG CẦN ML):")
    print(f"{'='*60}")
    print(f"Feature method: {feature_method}")
    print(f"Compare method: {compare_method}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"{'='*60}")
    
    if is_match:
        print(f"✅ HAI ĐẦU NỐI GIỐNG NHAU (similarity > {threshold})")
    else:
        print(f"❌ HAI ĐẦU NỐI KHÁC NHAU (similarity < {threshold})")
    
    print(f"{'='*60}\n")
    
    # Hiển thị ảnh
    display1 = img1.copy()
    display2 = img2.copy()
    
    # Resize để hiển thị
    h1, w1 = display1.shape[:2]
    h2, w2 = display2.shape[:2]
    max_h = max(h1, h2)
    
    display1 = cv2.resize(display1, (int(w1*max_h/h1), max_h))
    display2 = cv2.resize(display2, (int(w2*max_h/h2), max_h))
    
    combined = np.hstack([display1, display2])
    
    # Vẽ kết quả
    result_text = "MATCH" if is_match else "DIFFERENT"
    result_color = (0, 255, 0) if is_match else (0, 0, 255)
    cv2.putText(combined, result_text, (combined.shape[1]//2 - 80, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3)
    cv2.putText(combined, f"Similarity: {similarity:.4f}", (combined.shape[1]//2 - 120, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Comparison Result', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return similarity, is_match

# ======================
# TÌM THRESHOLD TỐI ƯU
# ======================
def find_optimal_threshold(data_dir, feature_method='hsv_histogram', 
                          compare_method='cosine', roi_dict=None, interactive=True):
    """
    Tìm threshold tối ưu từ dataset
    
    Args:
        data_dir: thư mục chứa data/type1, data/type2, ...
        roi_dict: dictionary {image_path: (x, y, w, h)} hoặc None
        interactive: True = chọn ROI bằng chuột nếu chưa có
    """
    import os
    from itertools import combinations
    
    if roi_dict is None:
        roi_dict = {}
    
    # Load tất cả ảnh theo class
    classes = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = []
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    images.append(img_path)
            if len(images) > 0:
                classes[class_name] = images
    
    print(f"Found {len(classes)} classes")
    for name, imgs in classes.items():
        print(f"  {name}: {len(imgs)} images")
    
    # Tính similarity cho tất cả các cặp
    same_similarities = []
    diff_similarities = []
    
    print("\nĐang tính toán similarities...")
    
    # Same class pairs
    for class_name, images in classes.items():
        for img1, img2 in combinations(images, 2):
            # Get or select ROI
            if img1 not in roi_dict:
                print(f"\nChọn ROI cho: {img1}")
                roi_dict[img1] = select_or_input_roi(cv2.imread(img1), interactive)
            
            if img2 not in roi_dict:
                print(f"\nChọn ROI cho: {img2}")
                roi_dict[img2] = select_or_input_roi(cv2.imread(img2), interactive)
            
            if roi_dict[img1] is None or roi_dict[img2] is None:
                continue
            
            crop1 = crop_wire_region(cv2.imread(img1), roi_dict[img1], interactive=False)
            crop2 = crop_wire_region(cv2.imread(img2), roi_dict[img2], interactive=False)
            
            if crop1 is not None and crop2 is not None:
                feat1 = extract_color_features(crop1, method=feature_method)
                feat2 = extract_color_features(crop2, method=feature_method)
                sim = compare_features(feat1, feat2, method=compare_method)
                same_similarities.append(sim)
    
    # Different class pairs
    class_names = list(classes.keys())
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            images1 = classes[class_names[i]]
            images2 = classes[class_names[j]]
            
            # Lấy 1 cặp từ mỗi pair of classes
            for img1 in images1[:3]:  # Chỉ lấy 3 ảnh đầu để nhanh
                for img2 in images2[:3]:
                    # Get or select ROI
                    if img1 not in roi_dict:
                        print(f"\nChọn ROI cho: {img1}")
                        roi_dict[img1] = select_or_input_roi(cv2.imread(img1), interactive)
                    
                    if img2 not in roi_dict:
                        print(f"\nChọn ROI cho: {img2}")
                        roi_dict[img2] = select_or_input_roi(cv2.imread(img2), interactive)
                    
                    if roi_dict[img1] is None or roi_dict[img2] is None:
                        continue
                    
                    crop1 = crop_wire_region(cv2.imread(img1), roi_dict[img1], interactive=False)
                    crop2 = crop_wire_region(cv2.imread(img2), roi_dict[img2], interactive=False)
                    
                    if crop1 is not None and crop2 is not None:
                        feat1 = extract_color_features(crop1, method=feature_method)
                        feat2 = extract_color_features(crop2, method=feature_method)
                        sim = compare_features(feat1, feat2, method=compare_method)
                        diff_similarities.append(sim)
    
    same_similarities = np.array(same_similarities)
    diff_similarities = np.array(diff_similarities)
    
    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH SIMILARITY:")
    print(f"{'='*60}")
    print(f"Same class similarities:")
    print(f"  Min:  {same_similarities.min():.4f}")
    print(f"  Max:  {same_similarities.max():.4f}")
    print(f"  Mean: {same_similarities.mean():.4f}")
    print(f"  Std:  {same_similarities.std():.4f}")
    print()
    print(f"Different class similarities:")
    print(f"  Min:  {diff_similarities.min():.4f}")
    print(f"  Max:  {diff_similarities.max():.4f}")
    print(f"  Mean: {diff_similarities.mean():.4f}")
    print(f"  Std:  {diff_similarities.std():.4f}")
    print()
    
    # Tìm threshold tối ưu
    optimal_threshold = (same_similarities.min() + diff_similarities.max()) / 2
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # Tính accuracy
    correct_same = np.sum(same_similarities > optimal_threshold)
    correct_diff = np.sum(diff_similarities <= optimal_threshold)
    total = len(same_similarities) + len(diff_similarities)
    accuracy = (correct_same + correct_diff) / total * 100
    
    print(f"Accuracy với threshold này: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # Vẽ biểu đồ
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(same_similarities, bins=20, alpha=0.7, label='Same class', color='green')
    plt.hist(diff_similarities, bins=20, alpha=0.7, label='Different class', color='red')
    plt.axvline(optimal_threshold, color='blue', linestyle='--', 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Similarity Distribution ({feature_method} + {compare_method})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([same_similarities, diff_similarities], labels=['Same', 'Different'])
    plt.ylabel('Similarity')
    plt.title('Similarity Boxplot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('similarity_analysis.png')
    print("✅ Đã lưu biểu đồ: similarity_analysis.png")
    plt.close()
    
    return optimal_threshold, accuracy, roi_dict

# ======================
# REAL-TIME
# ======================
class SimpleReferenceChecker:
    """Checker đơn giản không cần ML"""
    def __init__(self, feature_method='hsv_histogram', 
                 compare_method='cosine', threshold=0.85):
        self.feature_method = feature_method
        self.compare_method = compare_method
        self.threshold = threshold
        self.reference_feature = None
        self.reference_image = None
        self.reference_roi = None
        self.is_set = False
    
    def set_reference(self, image_path, roi=None, interactive=True):
        img = cv2.imread(image_path)
        if img is None:
            return False, "Không thể đọc ảnh"
        
        crop = crop_wire_region(img, roi, interactive)
        if crop is None:
            return False, "Không có ROI"
        
        self.reference_feature = extract_color_features(crop, method=self.feature_method)
        self.reference_image = img.copy()
        self.reference_roi = roi
        self.is_set = True
        
        return True, "Đã lưu mẫu chuẩn"
    
    def check(self, image_path, roi=None, interactive=True):
        if not self.is_set:
            return None, "Chưa có mẫu chuẩn"
        
        img = cv2.imread(image_path)
        if img is None:
            return None, "Không thể đọc ảnh"
        
        crop = crop_wire_region(img, roi, interactive)
        if crop is None:
            return None, "Không có ROI"
        
        feature = extract_color_features(crop, method=self.feature_method)
        similarity = compare_features(self.reference_feature, feature, 
                                     method=self.compare_method)
        
        is_match = similarity > self.threshold
        return similarity, is_match

# ======================
# MAIN
# ======================
def main():
    import sys
    
    print("\n" + "="*60)
    print("SO SÁNH MÀU DÂY - KHÔNG CẦN ML (ROI VERSION)")
    print("="*60)
    print("1. So sánh 2 ảnh")
    print("2. Tìm threshold tối ưu từ dataset")
    print("3. So sánh với mẫu chuẩn")
    print("="*60)
    
    choice = input("Nhập lựa chọn (1/2/3): ").strip()
    
    if choice == "1":
        print("\nSo sánh 2 ảnh")
        img1 = input("Đường dẫn ảnh 1: ").strip()
        img2 = input("Đường dẫn ảnh 2: ").strip()
        
        use_mouse = input("Dùng chuột chọn ROI? (y/n, mặc định y): ").strip().lower() != 'n'
        
        print("\nChọn feature method:")
        print("  1. hsv_histogram (chi tiết nhất, khuyến nghị)")
        print("  2. mean_hsv (nhanh)")
        print("  3. color_moments (đơn giản)")
        feat_choice = input("Chọn (1/2/3, mặc định 1): ").strip() or "1"
        feature_methods = {'1': 'hsv_histogram', '2': 'mean_hsv', '3': 'color_moments'}
        feature_method = feature_methods.get(feat_choice, 'hsv_histogram')
        
        print("\nChọn compare method:")
        print("  1. cosine (khuyến nghị)")
        print("  2. euclidean")
        print("  3. correlation")
        comp_choice = input("Chọn (1/2/3, mặc định 1): ").strip() or "1"
        compare_methods = {'1': 'cosine', '2': 'euclidean', '3': 'correlation'}
        compare_method = compare_methods.get(comp_choice, 'cosine')
        
        threshold = float(input("Threshold (mặc định 0.85): ").strip() or "0.85")
        
        compare_two_images_simple(img1, img2, feature_method, compare_method, 
                                 threshold, interactive=use_mouse)
    
    elif choice == "2":
        print("\nTìm threshold tối ưu")
        data_dir = input("Đường dẫn thư mục data: ").strip()
        
        use_mouse = input("Dùng chuột chọn ROI? (y/n, mặc định y): ").strip().lower() != 'n'
        
        print("\nChọn feature method:")
        print("  1. hsv_histogram")
        print("  2. mean_hsv")
        print("  3. color_moments")
        feat_choice = input("Chọn (1/2/3, mặc định 1): ").strip() or "1"
        feature_methods = {'1': 'hsv_histogram', '2': 'mean_hsv', '3': 'color_moments'}
        feature_method = feature_methods.get(feat_choice, 'hsv_histogram')
        
        print("\nChọn compare method:")
        print("  1. cosine")
        print("  2. euclidean")
        print("  3. correlation")
        comp_choice = input("Chọn (1/2/3, mặc định 1): ").strip() or "1"
        compare_methods = {'1': 'cosine', '2': 'euclidean', '3': 'correlation'}
        compare_method = compare_methods.get(comp_choice, 'cosine')
        
        optimal_threshold, accuracy, roi_dict = find_optimal_threshold(
            data_dir, feature_method, compare_method, interactive=use_mouse)
        
        print(f"\n✅ Sử dụng threshold: {optimal_threshold:.4f}")
        print(f"✅ Accuracy ước tính: {accuracy:.2f}%")
    
    elif choice == "3":
        print("\nSo sánh với mẫu chuẩn")
        reference = input("Đường dẫn ảnh mẫu chuẩn: ").strip()
        test_img = input("Đường dẫn ảnh cần kiểm tra: ").strip()
        
        use_mouse = input("Dùng chuột chọn ROI? (y/n, mặc định y): ").strip().lower() != 'n'
        
        threshold = float(input("Threshold (mặc định 0.85): ").strip() or "0.85")
        
        checker = SimpleReferenceChecker(threshold=threshold)
        success, msg = checker.set_reference(reference, interactive=use_mouse)
        
        if success:
            print(f"✅ {msg}")
            similarity, is_match = checker.check(test_img, interactive=use_mouse)
            
            if similarity is not None:
                print(f"\nSimilarity: {similarity:.4f}")
                if is_match:
                    print("✅ KHỚP VỚI MẪU CHUẨN")
                else:
                    print("❌ KHÔNG KHỚP VỚI MẪU CHUẨN")
        else:
            print(f"❌ {msg}")

if __name__ == "__main__":
    main()