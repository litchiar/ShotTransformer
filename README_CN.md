一个开箱即用的用于识别视频镜头类型的工具，能够分析视频并识别出镜头景别(shot-size)、拍摄角度(shot-angle)和镜头运动(shot-motion)。

来自于Toward Unified and Quantitative Cinematic Shot Attribute Analysis （有部分优化）
https://doi.org/10.3390/electronics12194174
数据集来自The Anatomy of Video Editing: A Dataset and Benchmark Suite for AI-Assisted Video Editing (https://link.springer.com/chapter/10.1007/978-3-031-20074-8_12)

注意：请从 **Release** 页面下载预训练模型，并将其放置在 `pretrained_models` 目录下。

### 快速开始
使用 `sample.py` 脚本可以快速测试项目功能：

```python
python sample.py
```

该脚本会处理 `test_video` 目录下的两个测试视频，并生成对应的JSON结果文件。

### API 详解

#### 主要函数

1. `process_video(video_path, shot_labels=['shot-size', 'shot-angle', 'shot-motion'], use_cn=False)`
   - 功能：处理视频并识别镜头类型
   - 参数：
     - `video_path`: 视频文件路径
     - `shot_labels`: 要识别的镜头类型列表，默认为['shot-size', 'shot-angle', 'shot-motion']
     - `use_cn`: 是否使用中文输出，默认为False
   - 返回：包含识别结果的JSON数据

2. `clear_model()`
   - 功能：清除加载的模型，释放内存
   - 参数：无
   - 返回：无

### 示例代码

```python
from usage import process_video, clear_model

# 处理单个视频，使用中文输出
result = process_video("test_video/0bce4c24-4809-4cc6-9577-acc3169a5988.mp4", use_cn=True)
print(result)

# 清除模型，释放内存
clear_model()
```

## 输出说明

处理视频后，会在视频所在目录生成一个与视频同名的JSON文件，包含识别结果。例如：


```json
{
    "shot_size": "medium",
    "shot_angle": "eye-level",
    "shot_motion": "handheld"
}
```
或

```json
{
    "镜头景别": "中景",
    "拍摄角度": "平视",
    "镜头运动": "手持"
}
```

### 可能的识别结果

- 镜头景别：超广角、广角、中景、特写、大特写
- 拍摄角度：高角度、低角度、平视、俯视、航拍
- 镜头运动：固定、手持、倾斜/升降、缩放/移动、平移/推拉

## 模型说明

项目使用 ShotTransformerV1 模型进行镜头类型识别，预训练模型保存在 `pretrained_models` 目录下。模型通过 YAML 配置文件进行加载和配置，配置文件位于 `yaml` 目录下。

## 注意事项
1. 确保视频文件格式支持 (mp4, avi, mov, mkv)
2. 处理视频可能需要较大的内存，特别是同时处理多个视频时
3. 使用完毕后建议调用 `clear_model()` 释放内存
4. 首次运行时会自动加载模型，可能需要一些时间
