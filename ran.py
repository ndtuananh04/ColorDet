import cv2
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

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

    extension = 80
    min_x = max(0, min_x - extension)
    max_x = min_x + 30
    padding = 20
    min_y = max(0, min_y - padding)
    max_y = min(frame.shape[0], max_y + padding)

    width = max_x - min_x
    height = max_y - min_y

    return (min_x, min_y, width, height), metal_pins

def crop_wire_region(image):
    result = detect_metal_connectors(image)
    if result is None:
        return None
    (x, y, w, h), _ = result
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
                               threshold=0.85):
    """
    So sánh 2 ảnh connector đơn giản, không cần ML
    
    Args:
        feature_method: 'hsv_histogram', 'mean_hsv', 'color_moments'
        compare_method: 'cosine', 'euclidean', 'correlation'
        threshold: ngưỡng quyết định (similarity > threshold → giống)
    """
    # Đọc ảnh
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("❌ Không thể đọc ảnh!")
        return None
    
    # Crop vùng dây
    crop1 = crop_wire_region(img1)
    crop2 = crop_wire_region(img2)
    
    if crop1 is None or crop2 is None:
        print("❌ Không phát hiện được vùng dây trong một trong hai ảnh")
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
    
    # Vẽ crop region
    result1 = detect_metal_connectors(img1)
    result2 = detect_metal_connectors(img2)
    
    if result1 is not None:
        (x, y, w, h), _ = result1
        cv2.rectangle(display1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if result2 is not None:
        (x, y, w, h), _ = result2
        cv2.rectangle(display2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
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
def find_optimal_threshold(data_dir, feature_method='hsv_histogram', compare_method='cosine'):
    """
    Tìm threshold tối ưu từ dataset
    
    Args:
        data_dir: thư mục chứa data/type1, data/type2, ...
    """
    import os
    from itertools import combinations
    
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
            crop1 = crop_wire_region(cv2.imread(img1))
            crop2 = crop_wire_region(cv2.imread(img2))
            
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
                    crop1 = crop_wire_region(cv2.imread(img1))
                    crop2 = crop_wire_region(cv2.imread(img2))
                    
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
    
    return optimal_threshold, accuracy

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
        self.is_set = False
    
    def set_reference(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False, "Không thể đọc ảnh"
        
        crop = crop_wire_region(img)
        if crop is None:
            return False, "Không phát hiện được vùng dây"
        
        self.reference_feature = extract_color_features(crop, method=self.feature_method)
        self.reference_image = img.copy()
        self.is_set = True
        
        return True, "Đã lưu mẫu chuẩn"
    
    def check(self, image_path):
        if not self.is_set:
            return None, "Chưa có mẫu chuẩn"
        
        img = cv2.imread(image_path)
        if img is None:
            return None, "Không thể đọc ảnh"
        
        crop = crop_wire_region(img)
        if crop is None:
            return None, "Không phát hiện được vùng dây"
        
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
    print("SO SÁNH MÀU DÂY - KHÔNG CẦN ML")
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
        
        compare_two_images_simple(img1, img2, feature_method, compare_method, threshold)
    
    elif choice == "2":
        print("\nTìm threshold tối ưu")
        data_dir = input("Đường dẫn thư mục data: ").strip()
        
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
        
        optimal_threshold, accuracy = find_optimal_threshold(data_dir, feature_method, compare_method)
        
        print(f"\n✅ Sử dụng threshold: {optimal_threshold:.4f}")
        print(f"✅ Accuracy ước tính: {accuracy:.2f}%")
    
    elif choice == "3":
        print("\nSo sánh với mẫu chuẩn")
        reference = input("Đường dẫn ảnh mẫu chuẩn: ").strip()
        test_img = input("Đường dẫn ảnh cần kiểm tra: ").strip()
        threshold = float(input("Threshold (mặc định 0.85): ").strip() or "0.85")
        
        checker = SimpleReferenceChecker(threshold=threshold)
        success, msg = checker.set_reference(reference)
        
        if success:
            print(f"✅ {msg}")
            similarity, is_match = checker.check(test_img)
            
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