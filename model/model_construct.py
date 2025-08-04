import torch
import yaml
import os
from collections import OrderedDict
from tools.load_dataset import get_project_path
dataset_dict = {
    'AVE': {
        'shot-size': ['medium', 'wide', 'close-up', 'extreme-wide', 'extreme-close-up'],
        'shot-angle': ['eye-level', 'high-angle', 'low-angle', 'overhead', 'aerial'],
        'shot-motion': ['locked', 'handheld', 'tilt', 'zoom', 'pan'],
    }
}


def get_num_class(dataset_name, label_type):
    return len(dataset_dict[dataset_name][label_type])


def construct_model(model_name, model_params: dict, dataset_name, label_type):
    num_class = get_num_class(dataset_name, label_type)
    model_params = model_params.copy()
    model_params.update({
        'num_class': num_class
    })
    if model_name=="ShotTransformerV1":
        from model.ShotTransformer_v1 import ShotTransformer_v1
        return ShotTransformer_v1(**model_params),num_class
def load_model_with_yaml(input_yaml):
    res=yaml.safe_load(open(input_yaml,mode='r',encoding='utf-8'))
    model=construct_model(model_name=res['model'],model_params=res['model_params'],dataset_name=res['dataset_name'],label_type=res['label_type'])
    pretrained_model_paths=os.path.join(get_project_path(),'pretrained_models')
    weights=load_pretrained_model(os.path.join(pretrained_model_paths,f"{res['model']}_{res['dataset_name']}_{res['label_type']}"))
    if weights is not None:
        model[0].load_state_dict(state_dict=weights,strict=True)
        print(f"load {res['label_type']} model success")
    else:
        print(f"load {res['label_type']} failed")
    return model[0]
def load_pretrained_model(pretrained_model_path):
    if not os.path.exists(pretrained_model_path):
        return None
    if not os.path.exists(os.path.join(pretrained_model_path,'last.ckpt')):
        return  None
    pretrain_state_dict=torch.load(os.path.join(pretrained_model_path,'last.ckpt'))['state_dict']

    new_order_dict=OrderedDict()
    for key,value in pretrain_state_dict.items():
        new_key=key[6:]
        new_order_dict[new_key]=value
    return new_order_dict
