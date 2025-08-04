import os
import json
import random

json_root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'json')


def get_dataset_split(stage):
    assert stage in ['train', 'test', 'val']
    with open(os.path.join(json_root_path, 'dataset_split.json'), mode='r', encoding='utf-8') as rj:
        data = json.loads(rj.read())
    return data[stage]


def get_sequence_data(rank):
    assert type(rank) is int
    assert rank >= 3 and rank <= 8
    if rank <= 5:
        file_path = os.path.join(json_root_path, 'sequence_data.json')
    else:
        file_path = os.path.join(json_root_path, 'high_rank_sequence_data.json')
    with open(file_path, mode='r', encoding='utf-8') as rj:
        data = json.loads(rj.read())
    # 直接修改这里
    full_data=data[f'k{rank}']
    print(len(full_data))
    limit_count=6
    random.seed(0)
    for scene in full_data:
        if len(full_data[scene])>limit_count:
            full_data[scene]=random.sample(full_data[scene],6)#这里是为了减少一些工作量 否则我会炸的
    return full_data


def get_sequence_data_full(rank):
    assert type(rank) is int
    assert rank >= 3 and rank <= 8
    if rank <= 5:
        file_path = os.path.join(json_root_path, 'sequence_data.json')
    else:
        file_path = os.path.join(json_root_path, 'high_rank_sequence_data.json')
    with open(file_path, mode='r', encoding='utf-8') as rj:
        data = json.loads(rj.read())
    return data[f'k{rank}']

def get_project_path():
    return os.path.dirname(os.path.dirname(__file__))


def get_ave_annotation():
    with open(os.path.join(get_project_path(), 'json', 'annotations.json'), mode='r', encoding='utf-8') as rf:
        data = json.loads(rf.read())
    return data


if __name__ == '__main__':
    # for stage in ['train', 'test', 'val']:
    #     print(stage, len(get_dataset_split(stage)))
    for rank in range(3, 4):
        print(rank, len(get_sequence_data(rank)))
