# ShotTransformer
An out-of-the-box tool for identifying video shot types, capable of analyzing videos and recognizing shot sizes, shot angles, and shot motions.

Derived from "Toward Unified and Quantitative Cinematic Shot Attribute Analysis" (with some optimizations)
https://doi.org/10.3390/electronics12194174
Dataset from "The Anatomy of Video Editing: A Dataset and Benchmark Suite for AI-Assisted Video Editing" (https://link.springer.com/chapter/10.1007/978-3-031-20074-8_12)

Note: Please download the pretrained models from the Release and place them in the pretrained_models directory.
The directory structure is:
```
pretrained_models/
├── ShotTransformerV1_AVE_shot-angle/
├── ShotTransformerV1_AVE_shot-motion/
└── ShotTransformerV1_AVE_shot-size/
```

## Quick Start
Use the `sample.py` script to quickly test the project functionality:

```python
python sample.py
```

This script will process the two test videos in the `test_video` directory and generate corresponding JSON result files.

## API Reference

1. `process_video(video_path, shot_labels=['shot-size', 'shot-angle', 'shot-motion'], use_cn=False)`
   - Function: Process video and identify shot types
   - Parameters:
     - `video_path`: Path to the video file
     - `shot_labels`: List of shot types to identify, default is ['shot-size', 'shot-angle', 'shot-motion']
     - `use_cn`: Whether to use Chinese output, default is False
   - Returns: JSON data containing recognition results

2. `clear_model()`
   - Function: Clear the loaded model and release memory
   - Parameters: None
   - Returns: None

### Example Code

```python
from usage import process_video, clear_model

# Process a single video with Chinese output
result = process_video("test_video/0bce4c24-4809-4cc6-9577-acc3169a5988.mp4", use_cn=True)
print(result)

# Clear the model and release memory
clear_model()
```

## Output Explanation

After processing a video, a JSON file with the same name as the video will be generated in the same directory, containing the recognition results. For example:

```json
{
    "shot_size": "medium",
    "shot_angle": "eye-level",
    "shot_motion": "handheld"
}
```
or

```json
{
    "镜头景别": "中景",
    "拍摄角度": "平视",
    "镜头运动": "手持"
}
```

### Possible Recognition Results

- Shot Size: Ultra-wide, Wide, Medium, Close-up, Extreme Close-up
- Shot Angle: High Angle, Low Angle, Eye Level, Bird's Eye View, Aerial View
- Shot Motion: Static, Handheld, Tilt/Crane, Zoom/Dolly, Pan/Tracking

## Model Description

The project uses the ShotTransformerV1 model for shot type recognition. Pretrained models are stored in the `pretrained_models` directory. Models are loaded and configured through YAML configuration files located in the `yaml` directory.

## Notes
1. Ensure video file formats are supported (mp4, avi, mov, mkv)
2. Processing videos may require significant memory, especially when processing multiple videos simultaneously
3. It is recommended to call `clear_model()` after use to release memory
4. The first run will automatically load the model, which may take some time