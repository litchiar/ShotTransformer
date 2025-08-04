from collections import OrderedDict

import torch
from einops import rearrange
from torch import nn
import timm
from slowfast.models.video_model_builder import MViT
from slowfast.config.defaults import get_cfg
from model.cfg_files import cfg_files
import torch.nn.functional as F
from model.R3D import R3DNet

def load_config(path_to_config):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if path_to_config is not None:
        cfg.merge_from_file(path_to_config)

    return cfg


class MotionTransformer(nn.Module):
    def __init__(self,motion_backbone, pretrain=False):
        super(MotionTransformer, self).__init__()
        assert motion_backbone in ['mvit','r3d']
        if motion_backbone=='mvit':
            cfg_file = cfg_files['MViTv2']
            self.cfg = load_config(cfg_file)
            self.model = MViT(self.cfg, num_class=1000)
            self.model.head = nn.Identity()
            # if self.pretrain:
            #     if len(self.pretrained_model) > 0:
            #         print('load pretrain', self.pretrained_model)
            #         checkpoint = torch.load(self.pretrained_model)
            #         new_state_dict = OrderedDict()
            #         for k, v in checkpoint['model_state'].items():
            #             if 'head' in k:
            #                 continue
            #             else:
            #                 new_state_dict[k] = v
            #         self.model.load_state_dict(new_state_dict, strict=False)
            #     else:
            #         print('missing pretrain model', self.pretrained_model)
        elif motion_backbone=='r3d':
            self.model=R3DNet(layer_sizes=(2, 2, 2, 2))

    def forward(self, x):
        return self.model(x)


class SimpleFrameDifference(nn.Module):
    # 直接计算差分
    def __init__(self, num_frames):
        super(SimpleFrameDifference, self).__init__()
        self.num_frames = num_frames

    def forward(self, frames):
        # 计算帧差分
        # frames: 输入的帧序列, shape为 (batch_size, num_channels, num_frames, height, width)
        # 提取前一帧和当前帧
        B, C, T, H, W = frames.shape
        prev_frame = frames[:, :, :-1, :, :]
        current_frame = frames[:, :, 1:, :, :]
        # 计算帧差分
        frame_diff = current_frame - prev_frame

        return frame_diff


class FrameDifferenceModule(nn.Module):
    # 来自论文 MoLo: Motion-augmented Long-short Contrastive Learning for Few-shot Action Recognition
    # 是论文STM: SpatioTemporal and Motion Encoding for Action Recognition 的改版
    def __init__(self, num_frames, in_channels=3, out_channels=3, kernel_size=3):
        super(FrameDifferenceModule, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                                    for _ in range(1, num_frames)])

    def forward(self, frames):
        # 计算帧差分
        # frames: 输入的帧序列, shape为 (batch_size, num_frames, num_channels, height, width)
        frames = frames.transpose(1, 2)
        batch_size, num_frames, num_channels, height, width = frames.size()
        frame_diffs = []
        for t in range(1, num_frames):
            prev_frame = frames[:, t - 1, :, :, :]
            current_frame = frames[:, t, :, :, :]
            # 经过不共享权重的卷积层并保持大小不变
            current_frame_conv = self.convs[t - 1](current_frame)
            # 帧差分
            frame_diff = current_frame_conv - prev_frame
            frame_diffs.append(frame_diff)
        # 拼接帧差分结果
        frame_diffs = torch.stack(frame_diffs, dim=1)

        return frame_diffs.transpose(1, 2)


class FrameDifferenceModule_Opt(nn.Module):
    def __init__(self, num_frames, individual_frame=True):
        super(FrameDifferenceModule_Opt, self).__init__()
        self.project = nn.Conv2d(in_channels=num_frames - 1, out_channels=num_frames - 1, kernel_size=(3, 3), padding=1,
                                 stride=1, groups=num_frames - 1 if individual_frame else 1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        prev = x[:, :, :-1, :, :]
        after = rearrange(x[:, :, 1:, :, :], 'b c t h w->(b c) t h w')
        after = rearrange(self.project(after), '(b c) t h w->b c t h w', b=B)
        difference = after - prev
        return difference


class MotionBranch(nn.Module):
    def __init__(self, fixed , difference,input_frame,motion_backbone, output_dim=512, fixed_frame=16,backbone_pretrain=True):
        '''
        :param fixed: 使用固定帧数的方法
        :param difference: 使用的差分方法
        :param input_frame: 真实的输入帧数
        :param fixed_frame: 固定的帧数
        :param output_dim: 输出的维度
        '''
        super(MotionBranch, self).__init__()
        assert fixed in ['interpolate','deconvolution']# 差值还是反卷积
        assert difference in ['frame_diff', 'conv_diff', 'conv_diff_opt','identity'] # 直接相减，卷积相减，优化版卷积相减，不相减直接输出
        self.difference = difference
        if self.difference == 'frame_diff':
            self.df = SimpleFrameDifference(num_frames=input_frame)
        elif self.difference == 'conv_diff':
            self.df = FrameDifferenceModule(num_frames=input_frame)
        elif self.difference == 'conv_diff_opt':
            self.df = FrameDifferenceModule_Opt(num_frames=input_frame)
        elif self.difference=='identity':
            self.df=nn.Identity()
        self.fixed=fixed
        if self.fixed == 'interpolate':
            self.ff = InterpolateFixed(fixed_frame=fixed_frame)
        self.motion_transformer=MotionTransformer(motion_backbone=motion_backbone,pretrain=backbone_pretrain)
        self.output_dim=output_dim
        if motion_backbone=='mvit':
            if self.output_dim !=768:
                self.use_proj=True
                self.proj=nn.Linear(768,self.output_dim)
            else:
                self.use_proj=False

        elif motion_backbone=='r3d':
            if self.output_dim != 512:
                self.use_proj = True
                self.proj = nn.Linear(512, self.output_dim)
            else:
                self.use_proj=False

    def forward(self, x):
        diff=self.df(x)
        feature=self.ff(diff)
        feature=self.motion_transformer(feature)
        if self.use_proj:
            feature=self.proj(feature)

        return feature

class InterpolateFixed(nn.Module):
    def __init__(self,fixed_frame=16,fixed_size=224):
        super(InterpolateFixed, self).__init__()
        self.fixed_frame=fixed_frame
        self.fixed_size=fixed_size
    def forward(self,x):
        return F.interpolate(x,size=(self.fixed_frame,self.fixed_size,self.fixed_size))


class StaticBranch(nn.Module):
    def __init__(self,static_backbone, output_feature_dim, pretrain=True):
        super(StaticBranch, self).__init__()
        self.output_dim = output_feature_dim
        assert static_backbone in ['res50','vit']
        if static_backbone=='vit':
            self.backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrain)  #
            self.backbone.head = nn.Linear(in_features=self.backbone.head.in_features, out_features=output_feature_dim)
        elif static_backbone=='res50':
            self.backbone = timm.create_model('resnet50d.a1_in1k', pretrained=pretrain)
            self.backbone.fc = nn.Linear(in_features=self.backbone.fc.in_features, out_features=output_feature_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        return self.backbone(x)


# End2End 动态与静态
class ShotTransformer_v1(nn.Module):
    def __init__(self, static_backbone,motion_backbone,num_class, branch,fixed, difference, static_branch_type, hidden_dim,input_frame,pretrain=False):
        super(ShotTransformer_v1, self).__init__()
        assert len(branch) >= 1
        self.branch = branch
        assert static_branch_type in ['start', 'middle', 'end']
        self.difference = difference
        self.static_branch_type = static_branch_type
        self.static_backbone=static_backbone
        self.motion_backbone=motion_backbone
        if 'static' in branch:
            self.static_branch = StaticBranch(static_backbone=static_backbone,output_feature_dim=hidden_dim,pretrain=pretrain)
        if 'motion' in branch:
            self.motion_branch=MotionBranch(motion_backbone=motion_backbone,fixed=fixed,difference=difference,input_frame=input_frame,output_dim=hidden_dim,backbone_pretrain=pretrain)
        self.num_class = num_class
        if len(branch)>1:
            self.ms_weight=nn.Parameter(torch.Tensor([0.5,0.5]))

        self.predict_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 256),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        #print(self.training)
        if 'static' in self.branch:
            if self.static_branch_type == 'start':
                static_img = x[:, :, 0, :, :]
            elif self.static_branch_type == 'middle':
                static_img = x[:, :, T // 2, :, :]
            else:
                static_img = x[:, :, -1, :, :]

        if len(self.branch) == 2:
            static_result=self.static_branch(static_img)
            motion_result=self.motion_branch(x)

            result=self.ms_weight[0]*static_result+self.ms_weight[1]*motion_result
            result=self.predict_head(result)
        elif len(self.branch) == 1:
            if 'static' in self.branch:
                static_result = self.static_branch(static_img)
                result = self.predict_head(static_result)
            elif 'motion' in self.branch:
                motion_result = self.motion_branch(x)
                result=self.predict_head(motion_result)
        return result


