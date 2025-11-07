"""
ColorExtractorPro Module
========================
Mô-đun trích xuất màu dây điện công nghiệp chính xác cao
Sử dụng: Hệ thống kiểm tra thứ tự màu dây điện tự động

Author: AI Assistant
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from scipy.spatial.distance import mahalanobis


@dataclass
class WirePosition:
    """Thông tin vị trí và hướng của một chân dây"""
    id: int                    # ID của chân dây
    x: int                     # Tọa độ x tâm chân dây
    y: int                     # Tọa độ y tâm chân dây
    angle: float               # Góc hướng dây (độ, 0° = ngang phải)
    connector_distance: int    # Khoảng cách tính từ đầu connector (px)


@dataclass
class ColorResult:
    """Kết quả nhận diện màu"""
    id: int                    # ID chân dây
    color: str                 # Tên màu nhận diện được
    confidence: float          # Độ tin cậy (0-1)
    bgr_value: Tuple[int, int, int]  # Giá trị BGR đại diện


class ColorStandard:
    """Lớp quản lý mẫu màu chuẩn trong không gian Lab"""
    
    def __init__(self):
        """Khởi tạo các mẫu màu chuẩn với mean và covariance matrix"""
        # Định nghĩa màu chuẩn trong không gian Lab
        # Format: 'tên_màu': {'mean': [L, a, b], 'cov': [[...], [...], [...]]}
        self.standards = {
            'white': {
                'mean': np.array([90.0, 0.0, 0.0]),
                'cov': np.array([[25.0, 0.0, 0.0],
                                [0.0, 4.0, 0.0],
                                [0.0, 0.0, 4.0]])
            },
            'black': {
                'mean': np.array([15.0, 0.0, 0.0]),
                'cov': np.array([[16.0, 0.0, 0.0],
                                [0.0, 4.0, 0.0],
                                [0.0, 0.0, 4.0]])
            },
            'red': {
                'mean': np.array([45.0, 55.0, 35.0]),
                'cov': np.array([[100.0, 10.0, 5.0],
                                [10.0, 64.0, 20.0],
                                [5.0, 20.0, 49.0]])
            },
            'green': {
                'mean': np.array([50.0, -45.0, 30.0]),
                'cov': np.array([[100.0, -10.0, 8.0],
                                [-10.0, 64.0, -15.0],
                                [8.0, -15.0, 49.0]])
            },
            'blue': {
                'mean': np.array([40.0, 10.0, -45.0]),
                'cov': np.array([[100.0, 5.0, -10.0],
                                [5.0, 36.0, -8.0],
                                [-10.0, -8.0, 64.0]])
            },
            'yellow': {
                'mean': np.array([85.0, -5.0, 75.0]),
                'cov': np.array([[64.0, -2.0, 15.0],
                                [-2.0, 16.0, -5.0],
                                [15.0, -5.0, 100.0]])
            },
            'orange': {
                'mean': np.array([65.0, 30.0, 55.0]),
                'cov': np.array([[81.0, 12.0, 18.0],
                                [12.0, 49.0, 15.0],
                                [18.0, 15.0, 81.0]])
            },
            'brown': {
                'mean': np.array([35.0, 15.0, 20.0]),
                'cov': np.array([[64.0, 8.0, 10.0],
                                [8.0, 25.0, 12.0],
                                [10.0, 12.0, 36.0]])
            },
            'gray': {
                'mean': np.array([55.0, 0.0, 0.0]),
                'cov': np.array([[64.0, 0.0, 0.0],
                                [0.0, 9.0, 0.0],
                                [0.0, 0.0, 9.0]])
            }
        }
        
        # Tính inverse covariance matrix cho mỗi màu (dùng cho Mahalanobis)
        for color_name in self.standards:
            self.standards[color_name]['cov_inv'] = np.linalg.inv(
                self.standards[color_name]['cov']
            )
    
    def get_color_names(self) -> List[str]:
        """Lấy danh sách tên màu chuẩn"""
        return list(self.standards.keys())
    
    def get_color_data(self, color_name: str) -> Dict:
        """Lấy dữ liệu màu chuẩn theo tên"""
        return self.standards.get(color_name, None)


class ColorExtractorPro:
    """
    Trích xuất màu dây điện công nghiệp chính xác cao
    
    Pipeline xử lý:
    1. White balance dựa trên vùng nhựa trắng connector
    2. Lọc nhiễu (bilateral/Gaussian)
    3. Tăng tương phản (CLAHE)
    4. Trích xuất ROI theo hướng dây
    5. Lọc outlier và tính median
    6. So sánh với mẫu chuẩn bằng Mahalanobis distance
    7. Lọc ổn định qua nhiều frame
    """
    
    def __init__(self,
                 roi_width: int = 10,
                 roi_height: int = 40,
                 roi_offset: int = 12,
                 outlier_percentile: float = 0.15,
                 mahalanobis_threshold: float = 3.5,
                 temporal_frames: int = 5,
                 use_clahe: bool = True,
                 filter_type: str = 'bilateral'):
        """
        Khởi tạo ColorExtractorPro
        
        Args:
            roi_width: Độ rộng ROI vuông góc với hướng dây (px)
            roi_height: Độ dài ROI theo hướng dây (px)
            roi_offset: Khoảng cách ROI từ đầu connector (px)
            outlier_percentile: Tỷ lệ pixel loại bỏ ở 2 đầu (0-0.5)
            mahalanobis_threshold: Ngưỡng khoảng cách Mahalanobis
            temporal_frames: Số frame dùng để lọc ổn định
            use_clahe: Có dùng CLAHE hay không
            filter_type: Loại filter ('bilateral' hoặc 'gaussian')
        """
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.roi_offset = roi_offset
        self.outlier_percentile = outlier_percentile
        self.mahalanobis_threshold = mahalanobis_threshold
        self.temporal_frames = temporal_frames
        self.use_clahe = use_clahe
        self.filter_type = filter_type
        
        # Khởi tạo mẫu màu chuẩn
        self.color_standard = ColorStandard()
        
        # Buffer lưu kết quả các frame gần nhất cho mỗi wire ID
        self.temporal_buffer: Dict[int, deque] = {}
        
        # CLAHE transformer
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # White balance reference (sẽ được cập nhật khi có white region)
        self.wb_reference: Optional[np.ndarray] = None
    
    def set_white_balance_region(self, image: np.ndarray, 
                                  white_region: Tuple[int, int, int, int]) -> None:
        """
        Thiết lập vùng tham chiếu cân bằng trắng
        
        Args:
            image: Ảnh đầu vào BGR
            white_region: (x, y, width, height) của vùng nhựa trắng
        """
        x, y, w, h = white_region
        white_roi = image[y:y+h, x:x+w]
        
        # Tính giá trị trung bình của vùng trắng
        mean_bgr = cv2.mean(white_roi)[:3]
        self.wb_reference = np.array(mean_bgr)
    
    def _apply_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng cân bằng trắng dựa trên vùng tham chiếu
        
        Args:
            image: Ảnh đầu vào BGR
            
        Returns:
            Ảnh đã cân bằng trắng
        """
        if self.wb_reference is None:
            return image.copy()
        
        # Tính hệ số điều chỉnh cho mỗi kênh
        # Giả sử giá trị lý tưởng của trắng là (255, 255, 255)
        target = np.array([255.0, 255.0, 255.0])
        scale_factors = target / (self.wb_reference + 1e-6)
        
        # Giới hạn scale factor để tránh quá điều chỉnh
        scale_factors = np.clip(scale_factors, 0.5, 2.0)
        
        # Áp dụng điều chỉnh
        balanced = image.astype(np.float32) * scale_factors
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        
        return balanced
    
    def _apply_noise_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng lọc nhiễu
        
        Args:
            image: Ảnh đầu vào BGR
            
        Returns:
            Ảnh đã lọc nhiễu
        """
        if self.filter_type == 'bilateral':
            # Bilateral filter: giữ cạnh tốt, loại nhiễu hiệu quả
            filtered = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
        else:  # gaussian
            filtered = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        return filtered
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng CLAHE để tăng tương phản
        
        Args:
            image: Ảnh đầu vào BGR
            
        Returns:
            Ảnh đã tăng tương phản
        """
        if not self.use_clahe:
            return image
        
        # Chuyển sang Lab, apply CLAHE lên kênh L
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        l_channel = self.clahe.apply(l_channel)
        
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _extract_roi(self, image: np.ndarray, 
                     wire_pos: WirePosition) -> Optional[np.ndarray]:
        """
        Trích xuất ROI theo hướng dây
        
        Args:
            image: Ảnh đã tiền xử lý
            wire_pos: Thông tin vị trí và hướng dây
            
        Returns:
            ROI đã trích xuất hoặc None nếu ngoài biên
        """
        h, w = image.shape[:2]
        
        # Tính điểm bắt đầu ROI (cách connector một khoảng roi_offset)
        angle_rad = np.deg2rad(wire_pos.angle)
        start_x = wire_pos.x + self.roi_offset * np.cos(angle_rad)
        start_y = wire_pos.y + self.roi_offset * np.sin(angle_rad)
        
        # Tạo các điểm góc của ROI (hình chữ nhật xoay)
        # Vector theo hướng dây
        dir_x = np.cos(angle_rad)
        dir_y = np.sin(angle_rad)
        
        # Vector vuông góc
        perp_x = -dir_y
        perp_y = dir_x
        
        # 4 góc của ROI
        half_w = self.roi_width / 2.0
        corners = np.array([
            [start_x - half_w * perp_x, start_y - half_w * perp_y],
            [start_x + half_w * perp_x, start_y + half_w * perp_y],
            [start_x + half_w * perp_x + self.roi_height * dir_x,
             start_y + half_w * perp_y + self.roi_height * dir_y],
            [start_x - half_w * perp_x + self.roi_height * dir_x,
             start_y - half_w * perp_y + self.roi_height * dir_y]
        ], dtype=np.float32)
        
        # Điểm đích (hình chữ nhật thẳng)
        dst_corners = np.array([
            [0, 0],
            [self.roi_width, 0],
            [self.roi_width, self.roi_height],
            [0, self.roi_height]
        ], dtype=np.float32)
        
        # Tính ma trận perspective transform
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        
        # Trích xuất ROI
        try:
            roi = cv2.warpPerspective(image, M, (self.roi_width, self.roi_height))
            return roi
        except:
            return None
    
    def _compute_representative_color(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Tính màu đại diện từ ROI bằng cách loại outlier và lấy median
        
        Args:
            roi: ROI đã trích xuất
            
        Returns:
            Giá trị BGR đại diện hoặc None
        """
        if roi is None or roi.size == 0:
            return None
        
        # Reshape thành danh sách các pixel
        pixels = roi.reshape(-1, 3).astype(np.float32)
        
        # Tính độ sáng (brightness) của mỗi pixel
        brightness = np.mean(pixels, axis=1)
        
        # Loại bỏ outlier: bỏ percentile thấp và cao nhất
        lower_bound = np.percentile(brightness, self.outlier_percentile * 100)
        upper_bound = np.percentile(brightness, (1 - self.outlier_percentile) * 100)
        
        # Lọc pixel
        mask = (brightness >= lower_bound) & (brightness <= upper_bound)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            return None
        
        # Tính median của mỗi kênh BGR
        representative_bgr = np.median(filtered_pixels, axis=0)
        
        return representative_bgr
    
    def _bgr_to_lab(self, bgr: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi màu BGR sang Lab
        
        Args:
            bgr: Mảng BGR (B, G, R)
            
        Returns:
            Mảng Lab (L, a, b)
        """
        # Tạo ảnh 1x1 pixel
        pixel = np.uint8([[bgr]])
        lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
        return lab_pixel[0, 0].astype(np.float32)
    
    def _compute_mahalanobis_distance(self, lab_value: np.ndarray,
                                       color_name: str) -> float:
        """
        Tính khoảng cách Mahalanobis giữa màu quan sát và màu chuẩn
        
        Args:
            lab_value: Giá trị Lab của màu quan sát
            color_name: Tên màu chuẩn
            
        Returns:
            Khoảng cách Mahalanobis
        """
        color_data = self.color_standard.get_color_data(color_name)
        if color_data is None:
            return float('inf')
        
        mean = color_data['mean']
        cov_inv = color_data['cov_inv']
        
        diff = lab_value - mean
        distance = np.sqrt(diff.T @ cov_inv @ diff)
        
        return distance
    
    def _classify_color(self, lab_value: np.ndarray) -> Tuple[str, float]:
        """
        Phân loại màu dựa trên khoảng cách Mahalanobis
        
        Args:
            lab_value: Giá trị Lab cần phân loại
            
        Returns:
            (tên_màu, confidence) hoặc ('unknown', 0.0)
        """
        min_distance = float('inf')
        best_color = 'unknown'
        
        # Tính khoảng cách đến tất cả màu chuẩn
        for color_name in self.color_standard.get_color_names():
            distance = self._compute_mahalanobis_distance(lab_value, color_name)
            
            if distance < min_distance:
                min_distance = distance
                best_color = color_name
        
        # Kiểm tra ngưỡng
        if min_distance > self.mahalanobis_threshold:
            return 'unknown', 0.0
        
        # Tính confidence (càng gần càng cao)
        confidence = max(0.0, 1.0 - min_distance / self.mahalanobis_threshold)
        
        return best_color, confidence
    
    def _temporal_filter(self, wire_id: int, color: str) -> str:
        """
        Lọc ổn định kết quả qua nhiều frame bằng mode
        
        Args:
            wire_id: ID của chân dây
            color: Màu nhận diện ở frame hiện tại
            
        Returns:
            Màu sau khi lọc ổn định
        """
        # Khởi tạo buffer nếu chưa có
        if wire_id not in self.temporal_buffer:
            self.temporal_buffer[wire_id] = deque(maxlen=self.temporal_frames)
        
        # Thêm kết quả hiện tại
        self.temporal_buffer[wire_id].append(color)
        
        # Nếu chưa đủ frame, trả về màu hiện tại
        if len(self.temporal_buffer[wire_id]) < self.temporal_frames:
            return color
        
        # Tính mode (giá trị xuất hiện nhiều nhất)
        colors = list(self.temporal_buffer[wire_id])
        from collections import Counter
        color_counts = Counter(colors)
        most_common_color = color_counts.most_common(1)[0][0]
        
        return most_common_color
    
    def process_frame(self, 
                      image: np.ndarray,
                      wire_positions: List[WirePosition],
                      white_region: Optional[Tuple[int, int, int, int]] = None,
                      visualize: bool = False) -> Tuple[List[ColorResult], Optional[np.ndarray]]:
        """
        Xử lý một frame để trích xuất màu các dây
        
        Args:
            image: Ảnh đầu vào BGR
            wire_positions: Danh sách vị trí và hướng các chân dây
            white_region: Vùng nhựa trắng để white balance (x, y, w, h)
            visualize: Có vẽ ROI lên ảnh không
            
        Returns:
            (results, visualization_image)
            - results: Danh sách ColorResult
            - visualization_image: Ảnh với ROI được vẽ (nếu visualize=True)
        """
        # 1. White balance
        if white_region is not None:
            self.set_white_balance_region(image, white_region)
        
        processed = self._apply_white_balance(image)
        
        # 2. Lọc nhiễu
        processed = self._apply_noise_filter(processed)
        
        # 3. Tăng tương phản (CLAHE)
        processed = self._apply_clahe(processed)
        
        # Chuẩn bị ảnh visualization nếu cần
        vis_image = None
        if visualize:
            vis_image = image.copy()
        
        # 4-7. Xử lý từng chân dây
        results = []
        
        for wire_pos in wire_positions:
            # Trích xuất ROI
            roi = self._extract_roi(processed, wire_pos)
            
            if roi is None:
                results.append(ColorResult(
                    id=wire_pos.id,
                    color='unknown',
                    confidence=0.0,
                    bgr_value=(0, 0, 0)
                ))
                continue
            
            # Tính màu đại diện
            repr_bgr = self._compute_representative_color(roi)
            
            if repr_bgr is None:
                results.append(ColorResult(
                    id=wire_pos.id,
                    color='unknown',
                    confidence=0.0,
                    bgr_value=(0, 0, 0)
                ))
                continue
            
            # Chuyển sang Lab
            repr_lab = self._bgr_to_lab(repr_bgr)
            
            # Phân loại màu
            color, confidence = self._classify_color(repr_lab)
            
            # Lọc ổn định qua nhiều frame
            filtered_color = self._temporal_filter(wire_pos.id, color)
            
            # Lưu kết quả
            results.append(ColorResult(
                id=wire_pos.id,
                color=filtered_color,
                confidence=confidence,
                bgr_value=tuple(repr_bgr.astype(int).tolist())
            ))
            
            # Vẽ ROI lên ảnh nếu cần
            if visualize and vis_image is not None:
                self._draw_roi(vis_image, wire_pos, filtered_color, confidence)
        
        return results, vis_image
    
    def _draw_roi(self, image: np.ndarray, wire_pos: WirePosition,
                  color: str, confidence: float) -> None:
        """
        Vẽ ROI và thông tin lên ảnh visualization
        
        Args:
            image: Ảnh để vẽ
            wire_pos: Vị trí chân dây
            color: Màu nhận diện
            confidence: Độ tin cậy
        """
        # Tính các điểm góc ROI
        angle_rad = np.deg2rad(wire_pos.angle)
        start_x = int(wire_pos.x + self.roi_offset * np.cos(angle_rad))
        start_y = int(wire_pos.y + self.roi_offset * np.sin(angle_rad))
        
        dir_x = np.cos(angle_rad)
        dir_y = np.sin(angle_rad)
        perp_x = -dir_y
        perp_y = dir_x
        
        half_w = self.roi_width / 2.0
        
        corners = [
            (int(start_x - half_w * perp_x), int(start_y - half_w * perp_y)),
            (int(start_x + half_w * perp_x), int(start_y + half_w * perp_y)),
            (int(start_x + half_w * perp_x + self.roi_height * dir_x),
             int(start_y + half_w * perp_y + self.roi_height * dir_y)),
            (int(start_x - half_w * perp_x + self.roi_height * dir_x),
             int(start_y - half_w * perp_y + self.roi_height * dir_y))
        ]
        
        # Vẽ hình chữ nhật ROI
        pts = np.array(corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        
        # Vẽ text thông tin
        text = f"{color} ({confidence:.2f})"
        text_pos = (corners[0][0], corners[0][1] - 5)
        cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    def reset_temporal_buffer(self, wire_id: Optional[int] = None) -> None:
        """
        Reset buffer lọc ổn định
        
        Args:
            wire_id: ID chân dây cần reset, None = reset tất cả
        """
        if wire_id is None:
            self.temporal_buffer.clear()
        elif wire_id in self.temporal_buffer:
            del self.temporal_buffer[wire_id]
    
    def export_results_dict(self, results: List[ColorResult]) -> List[Dict]:
        """
        Xuất kết quả dạng dictionary
        
        Args:
            results: Danh sách ColorResult
            
        Returns:
            Danh sách dictionary với format {id, color, confidence, bgr}
        """
        return [
            {
                'id': r.id,
                'color': r.color,
                'confidence': r.confidence,
                'bgr': r.bgr_value
            }
            for r in results
        ]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Ví dụ sử dụng ColorExtractorPro
    """
    
    # Khởi tạo extractor
    extractor = ColorExtractorPro(
        roi_width=10,
        roi_height=40,
        roi_offset=12,
        outlier_percentile=0.15,
        mahalanobis_threshold=3.5,
        temporal_frames=5,
        use_clahe=True,
        filter_type='bilateral'
    )
    
    # Giả lập ảnh đầu vào (thực tế sẽ từ camera)
    # image = cv2.imread('connector_image.jpg')
    image = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder
    image = cv2.imread("4.jpg")
    # Định nghĩa vị trí các chân dây (thực tế từ edge/corner detection)
    wire_positions = [
        WirePosition(id=1, x=100, y=200, angle=0, connector_distance=10),
        WirePosition(id=2, x=100, y=220, angle=0, connector_distance=10),
        WirePosition(id=3, x=100, y=240, angle=0, connector_distance=10),
        WirePosition(id=4, x=100, y=260, angle=0, connector_distance=10),
    ]
    
    # Vùng nhựa trắng của connector (x, y, width, height)
    white_region = (50, 180, 30, 100)
    
    # Xử lý frame
    results, vis_image = extractor.process_frame(
        image=image,
        wire_positions=wire_positions,
        white_region=white_region,
        visualize=True
    )
    
    # In kết quả
    print("=== Kết quả nhận diện màu dây ===")
    for result in results:
        print(f"ID: {result.id}, "
              f"Color: {result.color}, "
              f"Confidence: {result.confidence:.2f}, "
              f"BGR: {result.bgr_value}")
    
    # Xuất dạng dictionary
    results_dict = extractor.export_results_dict(results)
    print("\n=== Kết quả dạng dictionary ===")
    import json
    print(json.dumps(results_dict, indent=2))
    
    # Hiển thị ảnh visualization (nếu có)
    if vis_image is not None:
        cv2.imshow('Wire Color Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass