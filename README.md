# Wire Connector Order Inspection System  
<img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
<img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/OpenCV-4.5+-green.svg" alt="OpenCV">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">  

Automated quality control system for wire connector inspection in manufacturing. Uses Siamese Neural Networks to compare connector images and detect defects or wrong types. 
## Key Features:  
- Real-time connector verification (30+ FPS)
- Defect detection through similarity learning
- Multi-class classification (3-pin, 5-pin, 8-pin, etc.)
- Batch processing for quality audits
## Installation
1. Clone repository
```
git clone https://github.com/yourusername/ColorDet.git
cd ColorDet
```
2. Create virtual environment (Optional)
```
python -m venv venv
venv\Scripts\activate  (Windows)
source venv/bin/activate  (Linux)
```
3. Install dependencies
```pip install -r requirements.txt```
## Usage
1. Collect Training Data
```
python cap_image.py
```
Steps:
- Enter ROI size (e.g., 30x200)
- Create class folder (e.g., 5pin)
- Position ROI over connector
- Press SPACE to capture (50-100 images recommended)
- Press Q to finish  
2. Train Model
```
python siam_only_roi.py
```
3. Run Inference
```
python use.py
```
Options:
- Compare two images - Detailed visualization
- Real-time camera - Live inspection with ROI
# License
MIT License - see LICENSE file for details.
# Acknowledgments
- MobileNetV2 - torchvision
- Siamese Networks - Koch et al., 2015
- OpenCV - Computer vision tools
