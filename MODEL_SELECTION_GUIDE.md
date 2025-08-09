# Model Selection Guide

## Available Models

### YOLOv8 (Current)
- **File**: `best.pt` (your custom trained model)
- **Classes**: Custom floor plan classes (Column, Curtain Wall, Dimension, Door, Railing, Sliding Door, Stair Case, Wall, Window)
- **Best for**: Specific floor plan architectural detection
- **Status**: Your original trained model

### YOLOv11 (Higher Accuracy)
- **File**: `yolov11m.pt` (auto-downloaded)
- **Classes**: COCO dataset (80 classes)
- **Advantages**: 22% fewer parameters, higher mAP than YOLOv8
- **Best for**: General object detection with improved accuracy
- **Note**: Uses COCO classes, not your custom floor plan classes

### RT-DETR (Transformer)
- **File**: `rtdetr-l.pt` (auto-downloaded)
- **Classes**: COCO dataset (80 classes)
- **Advantages**: End-to-end transformer, NMS-free detection, 53.1% AP
- **Best for**: High-precision detection tasks
- **Note**: Uses COCO classes, suitable for furniture/object detection in floor plans

## Usage Recommendations

### For Floor Plan Elements (Walls, Doors, Windows)
- **Use**: YOLOv8 (Current) - Your custom trained model

### For Furniture Detection in Floor Plans
- **Use**: YOLOv11 or RT-DETR with classes like:
  - chair
  - dining table
  - couch
  - tv
  - bed
  - toilet
  - sink

### For Maximum Accuracy on General Objects
- **Use**: RT-DETR (Transformer)

### For Best Balance of Speed and Accuracy
- **Use**: YOLOv11 (Higher Accuracy)

## GPU Performance

All models support GPU acceleration and will automatically use CUDA if available. Expected performance improvements:
- YOLOv11: ~15-25% accuracy improvement over YOLOv8
- RT-DETR: Up to 108 FPS on modern GPUs
- Enhanced inference speed on all models with GPU

## Model Switching

The application automatically handles:
- Model downloading (first time only)
- GPU memory management
- Class label switching
- Session state management

Simply select a different model from the dropdown and click "Detect Objects" to switch models.
