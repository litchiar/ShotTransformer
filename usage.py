import os
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model.model_construct import dataset_dict
from tools.video_sample import sample_video
from model.model_construct import load_model_with_yaml
import gc
from tools.load_dataset import get_project_path
import tempfile
import shutil
import os
from pathlib import Path

models = dict()
SHOT_LANGUAGE = {'shot-size': {
    'cn': '镜头景别',
    'en': 'shot_size'
},
    'shot-angle': {
        'cn': '拍摄角度',
        'en': 'shot_angle'
    },
    'shot-motion': {
        'cn': '镜头运动',
        'en': 'shot_motion'
    },
    'description': {
        'cn': '描述',
        'en': 'description'
    }}  
CN_TRANS = {'shot-size': {
    "extreme-wide": "超广角",
    "wide": "广角",
    "medium": "中景",
    "close-up": "特写",
    "extreme-close-up": "大特写",

},
    'shot-angle': {
        "high-angle": "高角度",
        "low-angle": "低角度",
        "eye-level": "平视",
        "overhead": "俯视",
        "aerial": "航拍",

    },
    'shot-motion': {
        "locked": "固定",
        "handheld": "手持",
        "tilt": "倾斜/升降",
        "zoom": "缩放/移动",
        "pan": "平移/推拉",

    }
} 

def process_video(video_path, shot_labels=['shot-size', 'shot-angle', 'shot-motion'], use_cn=False):
    if not os.path.isfile(video_path) or not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return None
    language = 'en' if not use_cn else 'cn'
    temp_output_folder = tempfile.mkdtemp()
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir_path = os.path.dirname(video_path)
        video_output_folder = os.path.join(temp_output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        json_path = os.path.join(video_dir_path, f'{video_name}.json')
        if os.path.exists(json_path):
            print(f"skip: {video_path}")
            with open(json_path, 'r', encoding='utf-8') as jf:
                return json.load(jf)

        if not len(os.listdir(video_output_folder)) >= 4:
            sample_video(video_path, video_output_folder, frame=4)

        json_data = {}

        video_output_folder = Path(video_output_folder)
        image_paths = [str(img) for img in video_output_folder.glob("image_*.jpg")]
        if not len(image_paths) == 4:
            return None

        transform = transforms.Compose([transforms.ToTensor()])
        for label in shot_labels:
            if label not in models:
                models[label] = load_model_with_yaml(
                    os.path.join(get_project_path(), 'yaml', f'{label}.yml')).half().cuda()
                models[label].eval()

        image_tensors = []
        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).half().cuda()
            image_tensors.append(img_tensor)

        if image_tensors:
            batch_input = torch.cat(image_tensors, dim=0)
            batch_input = batch_input.unsqueeze(0)
            batch_input = batch_input.permute(0, 2, 1, 3, 4)

        for label, model in models.items():
            with torch.no_grad():
                outputs = model(batch_input)
            softmax_outputs = F.softmax(outputs, dim=1)
            max_indices = torch.argmax(softmax_outputs, dim=1)
            cine = dataset_dict['AVE'][label][
                max_indices.cpu().numpy().tolist()[0]]
            json_data[f'{SHOT_LANGUAGE[label][language]}'] = cine if language == 'en' else CN_TRANS[label][cine]

        with open(json_path, 'w', encoding='utf-8') as wj:
            json.dump(json_data, wj, indent=4, ensure_ascii=False)

        return json_data

    finally:
        shutil.rmtree(temp_output_folder, ignore_errors=True)


def clear_model():
    for label, model in models.items():
        models[label] = None  
    gc.collect()  #
    torch.cuda.empty_cache()  
   
