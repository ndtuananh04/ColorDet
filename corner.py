import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def detect_wires_hybrid(image_path, visualize=True):
    """
    Kết hợp Canny Edge Detection và Corner Detection để đếm số dây
    
    Args:
        image_path: đường dẫn đến ảnh đầu vào
        visualize: hiển thị kết quả trực quan
    
    Returns:
        wire_count: số lượng dây được phát hiện
        wire_positions: vị trí các dây
        confidence: độ tin cậy của kết quả
    """
    
    # ==================== ĐỌC VÀ TIỀN XỬ LÝ ẢNH ====================
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Làm mượt bằng Gaussian
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    print("=" * 60)
    print("🔍 BẮT ĐẦU PHÂN TÍCH ẢNH")
    print("=" * 60)
    print(f"Kích thước ảnh: {width}x{height}")
    
    # ==================== PHƯƠNG PHÁP 1: CANNY EDGE DETECTION ====================
    print("\n[1] CANNY EDGE DETECTION")
    print("-" * 60)
    
    # Bước 1-5 của Canny
    low_threshold = 30
    high_threshold = 100
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Morphology để làm sạch edges
    kernel = np.ones((2, 2), np.uint8)
    edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Phân tích vùng connector (60-80% bên phải)
    roi_start = int(width * 0.4)
    roi_end = int(width * 0.5)
    edge_roi = edges_clean[:, roi_start:roi_end]
    
    # Tính edge density theo trục Y
    edge_density = np.sum(edge_roi, axis=1) / 255
    edge_density_smooth = gaussian_filter1d(edge_density, sigma=3)
    
    # Tìm peaks (vị trí các dây)
    peaks_edges, properties_edges = find_peaks(
        edge_density_smooth,
        height=np.max(edge_density_smooth) * 0.15,
        distance=10,
        prominence=np.max(edge_density_smooth) * 0.1
    )
    
    wire_count_edges = len(peaks_edges)
    print(f"✓ Số dây phát hiện từ Canny: {wire_count_edges}")
    print(f"  Vị trí Y: {peaks_edges.tolist()}")
    
    # ==================== PHƯƠNG PHÁP 2: CORNER DETECTION ====================
    print("\n[2] HARRIS CORNER DETECTION")
    print("-" * 60)
    
    # Harris Corner Detection
    dst = cv2.cornerHarris(blurred, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    
    # Threshold để lấy corners mạnh
    threshold = 0.01 * dst.max()
    corners = np.argwhere(dst > threshold)
    
    # Lọc corners trong vùng ROI
    roi_corners = []
    for corner in corners:
        y, x = corner
        if roi_start < x < roi_end:
            roi_corners.append((x, y))
    
    print(f"✓ Tổng số corners phát hiện: {len(corners)}")
    print(f"  Corners trong vùng ROI: {len(roi_corners)}")
    
    # Clustering corners theo trục Y
    if len(roi_corners) > 0:
        y_coords = sorted([c[1] for c in roi_corners])
        
        # Nhóm corners gần nhau (cùng 1 dây)
        clusters = []
        current_cluster = [y_coords[0]]
        threshold_distance = 12
        
        for i in range(1, len(y_coords)):
            if y_coords[i] - current_cluster[-1] < threshold_distance:
                current_cluster.append(y_coords[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [y_coords[i]]
        clusters.append(current_cluster)
        
        # Tính vị trí trung bình mỗi cluster
        peaks_corners = [int(np.mean(cluster)) for cluster in clusters]
        wire_count_corners = len(peaks_corners)
        
        print(f"✓ Số dây phát hiện từ Corners: {wire_count_corners}")
        print(f"  Vị trí Y: {peaks_corners}")
    else:
        peaks_corners = []
        wire_count_corners = 0
        print("✗ Không phát hiện được corners trong ROI")
    
    # ==================== PHƯƠNG PHÁP 3: CONTOUR ANALYSIS ====================
    print("\n[3] CONTOUR ANALYSIS (BỔ SUNG)")
    print("-" * 60)
    
    # Tìm contours từ edges
    contours, _ = cv2.findContours(edge_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours hợp lệ
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if area > 30 and h > 3:
            center_y = y + h // 2
            valid_contours.append(center_y)
    
    # Nhóm contours gần nhau
    valid_contours = sorted(valid_contours)
    peaks_contours = []
    
    if len(valid_contours) > 0:
        current = valid_contours[0]
        peaks_contours.append(current)
        
        for y in valid_contours[1:]:
            if abs(y - current) > 12:
                peaks_contours.append(y)
                current = y
    
    wire_count_contours = len(peaks_contours)
    print(f"✓ Số dây phát hiện từ Contours: {wire_count_contours}")
    print(f"  Vị trí Y: {peaks_contours}")
    
    # ==================== KẾT HỢP KẾT QUẢ ====================
    print("\n[4] KẾT HỢP KẾT QUẢ")
    print("-" * 60)
    
    # Voting system: Kết hợp 3 phương pháp
    all_positions = []
    all_positions.extend([(p, 'edge') for p in peaks_edges])
    all_positions.extend([(p, 'corner') for p in peaks_corners])
    all_positions.extend([(p, 'contour') for p in peaks_contours])
    
    # Nhóm các vị trí gần nhau từ các phương pháp khác nhau
    all_y = sorted([p[0] for p in all_positions])
    
    final_positions = []
    votes_per_position = []
    
    if len(all_y) > 0:
        clusters = []
        current_cluster = [(all_y[0], [p[1] for p in all_positions if p[0] == all_y[0]])]
        
        for i in range(1, len(all_y)):
            if all_y[i] - current_cluster[-1][0] < 15:
                methods = [p[1] for p in all_positions if p[0] == all_y[i]]
                current_cluster.append((all_y[i], methods))
            else:
                clusters.append(current_cluster)
                current_cluster = [(all_y[i], [p[1] for p in all_positions if p[0] == all_y[i]])]
        clusters.append(current_cluster)
        
        # Tính điểm cho mỗi cluster
        for cluster in clusters:
            positions = [c[0] for c in cluster]
            methods = []
            for c in cluster:
                methods.extend(c[1])
            
            avg_position = int(np.mean(positions))
            vote_count = len(set(methods))  # Số phương pháp đồng ý
            
            final_positions.append(avg_position)
            votes_per_position.append(vote_count)
    
    wire_count_final = len(final_positions)
    
    # Tính confidence
    avg_votes = np.mean(votes_per_position) if votes_per_position else 0
    confidence = min(100, int((avg_votes / 3) * 100))
    
    print(f"✓ Số dây cuối cùng: {wire_count_final}")
    print(f"  Vị trí Y: {final_positions}")
    print(f"  Số phương pháp đồng ý mỗi vị trí: {votes_per_position}")
    print(f"  Độ tin cậy: {confidence}%")
    
    # ==================== TRỰC QUAN HÓA KẾT QUẢ ====================
    if visualize:
        visualize_results(
            img, gray, blurred, edges, edges_clean, dst, edge_density_smooth,
            peaks_edges, roi_corners, final_positions, votes_per_position,
            wire_count_final, confidence, roi_start, roi_end
        )
    
    return wire_count_final, final_positions, confidence

def visualize_results(img, gray, blurred, edges, edges_clean, dst, 
                      edge_density, peaks_edges, corners, final_positions, 
                      votes, wire_count, confidence, roi_start, roi_end):
    """Hiển thị kết quả trực quan"""
    
    height, width = gray.shape
    
    # Tạo ảnh kết quả với annotations
    img_result = img.copy()
    img_edges = cv2.cvtColor(edges_clean, cv2.COLOR_GRAY2BGR)
    img_corners = img.copy()
    
    # Vẽ ROI
    cv2.rectangle(img_result, (roi_start, 0), (roi_end, height), (255, 0, 0), 2)
    cv2.putText(img_result, 'ROI', (roi_start+5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Vẽ corners
    for corner in corners:
        x, y = corner
        cv2.circle(img_corners, (x, y), 2, (0, 0, 255), -1)
    
    # Vẽ vị trí dây cuối cùng
    for idx, (pos, vote) in enumerate(zip(final_positions, votes)):
        color = (0, 255, 0) if vote >= 2 else (0, 255, 255)
        thickness = 3 if vote >= 2 else 2
        
        cv2.line(img_result, (0, pos), (width, pos), color, thickness)
        cv2.circle(img_result, (roi_start + 20, pos), 6, (0, 0, 255), -1)
        cv2.putText(img_result, f'#{idx+1} ({vote}/3)', (10, pos-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Tạo figure
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Quá trình xử lý
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('1. Ảnh gốc', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('2. Gaussian Blur (σ=1.4)', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('3. Canny Edge (30-100)', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    # Row 2: Các phương pháp detection
    plt.subplot(3, 3, 4)
    plt.imshow(img_edges)
    for peak in peaks_edges:
        plt.axhline(y=peak, color='r', linewidth=2, alpha=0.6)
    plt.title(f'4. Edge Detection\n({len(peaks_edges)} wires)', 
              fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    plt.title(f'5. Corner Detection\n({len(corners)} corners)', 
              fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    corner_map = np.zeros_like(gray)
    corner_map[dst > 0.01 * dst.max()] = 255
    plt.imshow(corner_map, cmap='hot')
    plt.title('6. Harris Corner Response', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    # Row 3: Phân tích và kết quả
    plt.subplot(3, 3, 7)
    plt.plot(edge_density, range(len(edge_density)), 'b-', linewidth=2, alpha=0.7)
    plt.plot(edge_density[peaks_edges], peaks_edges, 'ro', markersize=10)
    plt.gca().invert_yaxis()
    plt.xlabel('Edge Density', fontsize=10)
    plt.ylabel('Y Position (pixels)', fontsize=10)
    plt.title('7. Edge Density Analysis', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 8)
    if len(votes) > 0:
        bars = plt.barh(range(len(votes)), votes, color=['green' if v>=2 else 'orange' for v in votes])
        plt.yticks(range(len(votes)), [f'Wire {i+1}' for i in range(len(votes))])
        plt.xlabel('Voting Score (max 3)', fontsize=10)
        plt.title('8. Confidence per Wire', fontsize=11, fontweight='bold')
        plt.xlim(0, 3)
        plt.grid(True, alpha=0.3, axis='x')
    
    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f'9. FINAL RESULT\n{wire_count} wires detected ({confidence}% confidence)', 
              fontsize=12, fontweight='bold', color='red')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== SỬ DỤNG ====================
if __name__ == "__main__":
    image_path = "4.jpg"
    
    try:
        wire_count, positions, confidence = detect_wires_hybrid(image_path, visualize=True)
        
        print("\n" + "=" * 60)
        print("📊 KẾT QUẢ CUỐI CÙNG")
        print("=" * 60)
        print(f"🔌 Số dây: {wire_count}")
        print(f"📍 Vị trí (pixels): {positions}")
        print(f"✅ Độ tin cậy: {confidence}%")
        print("=" * 60)
        
        if confidence >= 75:
            print("✓ Kết quả có độ tin cậy cao!")
        elif confidence >= 50:
            print("⚠ Kết quả khá tốt, nhưng nên kiểm tra lại")
        else:
            print("⚠ Kết quả có độ tin cậy thấp, cần điều chỉnh tham số")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("\n📝 Hướng dẫn:")
        print("1. pip install opencv-python numpy matplotlib scipy")
        print("2. Đặt tên ảnh là 'wire_connector.jpg' hoặc sửa đường dẫn")
        print("3. Điều chỉnh tham số nếu kết quả chưa chính xác")