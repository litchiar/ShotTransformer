import os
import json
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from tqdm import tqdm


def sample_video(video_path,output_dir,frame=16,width=224,height=224,frame_pos='middle'):
    '''
    :param video_path: 视频路径
    :param output_dir: 图片保存目录
    :param frame: 采样几帧一般16或者8
    :param width: 保存图片的宽度
    :param height: 保存图片的高度
    :return: 是否成功 如果镜头时长小于frame就返回False
    '''
    assert frame_pos in ['begin','middle', 'end']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(7))
    if frame_count < frame:
        # print(video_path,'frame_count <= seg_count')
        return False
    average_duration = frame_count / frame
    video_clip = [[round(i * average_duration), round(i * average_duration + average_duration - 1)] for i in
                  range(frame)]
    video_clip[-1][-1] = frame_count - 1
    for i, clip in enumerate(video_clip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip[0])
        res, img = cap.read()
        if frame_pos == 'begin':
            pass
        elif frame_pos == 'end':
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip[1])
            res, img = cap.read()
        elif frame_pos=='middle':
            cap.set(cv2.CAP_PROP_POS_FRAMES, (clip[0] + clip[1]) // 2)
            res, img = cap.read()
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        Image.fromarray(img).save(os.path.join(output_dir, f"image_{i}.jpg"))
    cap.release()
    return True

def calcu_mean_std(dataset:Dataset):#计算均值与方差
    mean = [0, 0, 0]
    std = [0, 0, 0]
    for i in tqdm(range(len(dataset))):
        imgs = dataset[i][0]
        label= dataset[i][1]
        for i in range(3):
            mean[i] += imgs[:, i, :, :].mean()
            std[i] += imgs[:, i, :, :].std()
    for i in range(3):
        mean[i] = mean[i] / len(dataset)
        std[i] = std[i] / len(dataset)
    return mean, std
def show_img(dataset:Dataset,output_dir,idx=0):#可视化
    topil=ToPILImage()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(dataset[idx][0]):
        topil(frame).save(os.path.join(output_dir, f'image_{i}.jpg'))
def get_label_set(dataset:Dataset): #获取数据集的标签类型
    labels=[]
    for i in tqdm(range(len(dataset))):
        if type(dataset[i][1]) is list:
            labels.extend(dataset[i][1])
        else:
            labels.append(dataset[i][1])
    return set(labels)
def get_dataset_alpha(dataset:Dataset): #获取数据集的标签类型
    labels=[]
    for i in tqdm(range(len(dataset))):
        if type(dataset[i][1]) is list:
            labels.extend(dataset[i][1])
        else:
            labels.append(dataset[i][1])
    from collections import Counter
    frequency=Counter(labels)
    total = sum(frequency.values())
    decimal_frequency = {num: round(count / total, 3) for num, count in frequency.items()}
    sorted_keys = sorted(decimal_frequency.keys())
    decimal_frequency_list = [decimal_frequency[key] for key in sorted_keys]
    return decimal_frequency_list
