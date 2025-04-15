# PPE Detection and Danger Zone Monitoring System

A real-time computer vision system that monitors construction sites to ensure workers are wearing proper Personal Protective Equipment (PPE) and are not entering defined danger zones.

## Recent Improvements (macos branch)

This branch introduces macOS support and several improvements:

- **macOS Support**: Full compatibility with macOS systems
- **Automatic Video Fallback**: If no camera is detected, the system automatically uses an available video file
- **Cross-Platform Compatibility**: Works seamlessly across Windows, Linux, and macOS
- **Improved Error Handling**: Better handling of missing cameras and video sources

## Recent Improvements (Hugo14april branch)

This branch introduces several significant improvements to the system:

- **Enhanced UI**: Modern interface with improved styling, icons, and better user feedback
- **Cross-Platform Support**: Complete compatibility with both Windows PCs and Jetson Nano devices
- **Video Source Selection**: New UI component to easily switch between camera and video files
- **Multiple Camera Support**: Dropdown menu to select from available cameras on the system
- **Jetson Nano Optimization**: Special camera handling with GStreamer for optimal performance on Jetson devices
- **Fixed Zone Management**: Improved zone creation and management with proper source-specific zones
- **Platform-Independent Paths**: Robust file handling across different operating systems
- **Command-Line Flexibility**: Added options to specify server IP and port for network accessibility

## Features

- **PPE Detection**: Automatically detects if workers are wearing helmets and safety vests
- **Danger Zone Monitoring**: Define custom danger zones and receive alerts when workers enter them
- **Multi-platform Support**: Works on Windows, Linux, and macOS systems, including NVIDIA Jetson Nano
- **Flexible Video Sources**: Use webcams, IP cameras, or pre-recorded videos
- **User-friendly Interface**: Modern web interface with real-time monitoring and alerts
- **Notification System**: Receive instant alerts for safety violations with screenshots
- **Statistics Dashboard**: Track safety metrics and compliance over time

## Requirements

- Python 3.7+ (Python 3.9 recommended)
- NVIDIA GPU recommended for faster inference (supports CPU mode as well)
- For Jetson Nano: JetPack 4.6+ with CUDA support
- For macOS: Python 3.8+ with OpenCV and PyTorch

See `requirements.txt` for Python dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PPE-Detection-and-Danger-Zone-Monitoring-System.git
cd PPE-Detection-and-Danger-Zone-Monitoring-System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv11 model files and place them in the `models` directory:
   - `yolov11s.pt` (person detection model)
   - `ppe_v11s.pt` (PPE detection model)

## Usage

### Basic Usage
```bash
python main.py
```

This starts the system using your default camera if available, or automatically falls back to an available video file if no camera is detected.

### With a Video File
```bash
python main.py --Input "path/to/your/video.mp4"
```

### Specify Server IP and Port
```bash
python main.py --ip 0.0.0.0 --port 5000
```
Using 0.0.0.0 allows access from other devices on the network.

### Platform-Specific Notes

- **Windows**: Uses DirectShow for camera access
- **Linux/Jetson Nano**: Uses V4L2 and GStreamer for optimized camera access
- **macOS**: Automatically falls back to video files if no camera is detected

## UI Instructions

1. **Home Page**: Shows the live PPE detection feed
2. **Manage Page**: Define danger zones by drawing polygons on the video feed
3. **Statistics Page**: View PPE compliance statistics and reports
4. **Screenshots Page**: Access captured safety violation incidents

## How It Works

1. The system processes video frames in real-time using YOLOv11 models
2. First, people are detected in the frame
3. Then, PPE items (helmets and vests) are detected and associated with each person
4. The system checks if people are:
   - Wearing proper PPE
   - Located inside defined danger zones
5. Alerts are generated for any safety violations

## Cross-Platform Support

- **Windows**: Uses DirectShow for camera access
- **Linux/Jetson Nano**: Uses V4L2 and GStreamer for optimized camera access
- **macOS**: Supports camera input with automatic fallback to video files
- **All Platforms**: Same web interface and detection capabilities

## Customization

- Adjust detection thresholds in the UI settings
- Draw custom danger zones specific to each video source
- Use multiple cameras or video sources

## Project Structure

- `/models` - YOLOv11 detection models
- `/static` - Web UI assets and captured screenshots
- `/templates` - HTML templates for the web interface
- `/video` - Sample videos for testing
- `/zone` - Saved danger zone configurations 