import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from ..backbone_2d_yolov7.yolo import YoloBody
from ..temporal_shift_module.ops.models import TSN

# Channel Self Attetion Module
class CSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out

class MyConv2d(nn.Module):
    def __init__(self, in_channel,out_channel,k,padding):
        super().__init__()
        convs = []
        convs.append(nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=k,
                               padding=padding))
        convs.append(nn.BatchNorm2d(out_channel))
        convs.append(nn.LeakyReLU(0.1, inplace=True))
        self.convs = nn.Sequential(*convs)
        
    def forward(self, x):
        x = self.convs(x)
        return x

# Channel Encoder
class ChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.SCAM = CSAM()
        self.fuse_convs = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            MyConv2d(out_dim, out_dim,k=3, padding=1),
            self.SCAM,
            MyConv2d(out_dim, out_dim,k=3, padding=1),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x1, x2):
        """
            x: [B, C, H, W]
        """
        x = torch.cat([x1, x2], dim=1)
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x

# You Only Watch Once
class YOWO(nn.Module):
    def __init__(self, 
                 num_classes = 24, 
                 trainable = False):
        super(YOWO, self).__init__()
        self.num_classes = num_classes
        self.trainable = trainable
        self.pretrained_2d = True
        self.pretrained_3d = True
        # ------------------ Network ---------------------
        ## 2D backbone
        # self.backbone_2d, bk_dim_2d = build_backbone_2d(
        #     cfg, pretrained=cfg['pretrained_2d'] and trainable)
        
        # self.backbone_2d = YoloBody(24, 's')
        anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.backbone_2d = YoloBody(anchors_mask=anchors_mask,num_classes=24)
        if self.pretrained_2d:
            url = 'https://github.com/bubbliiiing/yolov7-tiny-pytorch/releases/download/v1.0/yolov7_tiny_weights.pth'
            
            # checkpoint = torch.load('/root/autodl-tmp/YOWOv2_TSM_Pre/Pretrain/yolox_s.pth') #load_state_dict_from_url(url, map_location='cpu')
            # checkpoint_state_dict = checkpoint.pop('state_dict')
            # cp yolox_s.pth 

            # check
            if url is None:
                print('No 2D pretrained weight ...')
                return self.backbone_2d 
            else:
                print('Loading 2D backbone pretrained weight: {}'.format("YOLO-X"))

                # state dict
                checkpoint = load_state_dict_from_url(url, map_location='cpu')
                # checkpoint_state_dict = checkpoint.pop('model')
                checkpoint_state_dict = checkpoint

                # model state dict
                model_state_dict = self.backbone_2d.state_dict()
                # for k in model_state_dict.keys():
                #     print("model_state_dict",k)
                # check
                for k in list(checkpoint_state_dict.keys()):
                    if k in model_state_dict:
                        shape_model = tuple(model_state_dict[k].shape)
                        shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                        if shape_model != shape_checkpoint:
                            print("self.backbone_2d k in model_state_dict",k)
                            checkpoint_state_dict.pop(k)
                    else:
                        checkpoint_state_dict.pop(k)
                        print("self.backbone_2d k not in model_state_dict",k)

                self.backbone_2d.load_state_dict(checkpoint_state_dict, strict=False)
        
        self.backbone_3d = TSN(num_classes, num_segments=16, modality='RGB',
                                base_model='mobilenetv2',
                                dropout=0.5,
                                img_feature_dim=224,
                                partial_bn=True,
                                is_shift=True, shift_div=8, shift_place="blockres",
                                fc_lr5=True,
                                non_local=False)
        
        if self.pretrained_3d:
            print('Loading 3D backbone pretrained weight: {}'.format("TSM"))

            # state dict
            checkpoint = torch.load('/root/autodl-tmp/YOWOv2_TSM_yolov7/Pretrain/ckpt.best.pth.tar',map_location='cpu') #load_state_dict_from_url(url, map_location='cpu')
            checkpoint_state_dict = checkpoint.pop('state_dict')
            # checkpoint_state_dict = checkpoint
            # for k in checkpoint_state_dict.keys():
            #     print("model_state_dict",k)

            # model state dict
            model_state_dict = self.backbone_3d.state_dict()
            # for k in model_state_dict.keys():
            #     print("model_state_dict",k)
            # check

            # reformat checkpoint_state_dict:
            new_state_dict = {}
            for k in checkpoint_state_dict.keys():
                v = checkpoint_state_dict[k]
                new_state_dict[k[7:]] = v

            # check
            for k in list(new_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(new_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        new_state_dict.pop(k)
                        print("self.backbone_3d k in model_state_dict",k)
                else:
                    new_state_dict.pop(k)
                    print("self.backbone_3d k not in model_state_dict",k)

            self.backbone_3d.load_state_dict(new_state_dict, strict=False)
        
        self.ChannelEncoder1 = ChannelEncoder(1280+512,64)
        self.ChannelEncoder2 = ChannelEncoder(1280+256,64)
        self.ChannelEncoder3 = ChannelEncoder(1280+128,64)
        self.channel_feat = [self.ChannelEncoder1,
                             self.ChannelEncoder2,
                             self.ChannelEncoder3]
        self.yolo_head_P3 = nn.Conv2d(64, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(64, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(64, len(anchors_mask[0]) * (5 + num_classes), 1)
    
    def forward(self, video_clips):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        return:
            outputs: (Dict) -> {
                'pred_conf': (Tensor) [B, M, 1]
                'pred_cls':  (Tensor) [B, M, C]
                'pred_reg':  (Tensor) [B, M, 4]
                'anchors':   (Tensor) [M, 2]
                'stride':    (Int)
            }
        """                        
        # key frame
        key_frame = video_clips[:, :, -1, :, :]
        # 3D backbone
        feat_3d = self.backbone_3d(video_clips)
                    
        # print("feat_3d",feat_3d.size()) #feat_3d torch.Size([1, 1280, 7, 7])

        # 2D backbone
        levels_feats = self.backbone_2d(key_frame)
        
        level = 2
        P = []
        for level_feat in levels_feats:
            # upsample
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            # encoder
            p = self.channel_feat[2-level](feat_3d_up,level_feat)
            P.append(p)
            level = level-1
        out2 = self.yolo_head_P3(P[0])
        out1 = self.yolo_head_P4(P[1])
        out0 = self.yolo_head_P5(P[2])
            
        # torch.Size([1, 64, 7, 7])
        # torch.Size([1, 64, 14, 14])
        # torch.Size([1, 64, 28, 28])
        # out2 torch.Size([1, 87, 7, 7])
        # out1 torch.Size([1, 87, 14, 14])
        # out0 torch.Size([1, 87, 28, 28])

        return [out0,out1,out2]
