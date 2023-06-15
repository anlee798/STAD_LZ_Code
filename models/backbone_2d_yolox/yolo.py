#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

try:
    from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
except:
    from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        
        # non-shared heads
        all_reg_preds = []
        all_obj_preds = []
        all_cls_preds = []
        
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)
            
            #output      = torch.cat([reg_output, obj_output, cls_output], 1)
            reg_output = reg_output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            obj_output = obj_output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_output = cls_output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
           
            all_reg_preds.append(reg_output)
            all_obj_preds.append(obj_output)
            all_cls_preds.append(cls_output)
            #outputs.append(output)
        outputs = {"pred_reg": all_reg_preds,       # List(Tensor) [B, M, 4]
                       "pred_obj": all_obj_preds,         # List(Tensor) [B, M, 1]
                       "pred_cls": all_cls_preds,         # List(Tensor) [B, M, numclass]
                       }            # List(Int)
        return outputs

class YOLOXHead2(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 64, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 64, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  torch.Size([1, 128, 28, 28])
        #   P4_out  torch.Size([1, 256, 14, 14])
        #   P5_out  torch.Size([1, 512, 7, 7])
        #---------------------------------------------------#
        outputs = []
        
        # non-shared heads
        all_reg_preds = []
        all_cls_preds = []
        
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #print("x.shape",x.shape)    # x.shape torch.Size([1, 128, 28, 28])
                                        # x.shape torch.Size([1, 128, 14, 14])
                                        # x.shape torch.Size([1, 128, 7, 7])
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #print("cls_feat.shape",cls_feat.shape) #torch.Size([1, 128, 28, 28])
                                                    #torch.Size([1, 128, 14, 14])
                                                    #torch.Size([1, 128, 7, 7])
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)
            #print("cls_output.shape",cls_output.shape) #torch.Size([1, 24, 28, 28])
                                                        #torch.Size([1, 24, 14, 14])
                                                        #torch.Size([1, 24, 7, 7])

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #print("reg_feat.shape",reg_feat.shape)   #torch.Size([1, 128, 28, 28])
                                                    #torch.Size([1, 128, 14, 14])
                                                    #torch.Size([1, 128, 7, 7])
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            # print("reg_output.shape",reg_output.shape) # torch.Size([1, 4, 28, 28])
                                                        #torch.Size([1, 4, 14, 14])
                                                        #torch.Size([1, 4, 7, 7])
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            #obj_output  = self.obj_preds[k](reg_feat)
            
            #output      = torch.cat([reg_output, obj_output, cls_output], 1)
            #reg_output = reg_output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            #obj_output = obj_output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            #cls_output = cls_output.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
           
            all_reg_preds.append(reg_output)
            # all_obj_preds.append(obj_output)
            all_cls_preds.append(cls_output)
            #outputs.append(output)
        # outputs = {"pred_reg": all_reg_preds,       # List(Tensor) [B, M, 4]
        #                "pred_cls": all_cls_preds,         # List(Tensor) [B, M, numclass]
        #                }            # List(Int)
        
        return all_cls_preds, all_reg_preds

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample) 

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi, pretrained = False):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead2(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        # print("fpn_outs",fpn_outs[0].shape) # torch.Size([1, 128, 28, 28])
        # print("fpn_outs",fpn_outs[1].shape) # torch.Size([1, 256, 14, 14])
        # print("fpn_outs",fpn_outs[2].shape) # torch.Size([1, 512, 7, 7])
        outputs1,outputs2     = self.head.forward(fpn_outs)
        return outputs1,outputs2
    
def build_yolo_x(num_classes = 24,phi='s',pretrained = True):
    model = YoloBody(num_classes=num_classes,phi=phi)
    if pretrained:
        url = 'https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_s.pth'

        # check
        if url is None:
            print('No 2D pretrained weight ...')
            return model
        else:
            print('Loading 2D backbone pretrained weight: {}'.format("YOLO-X"))

            # state dict
            checkpoint = load_state_dict_from_url(url, map_location='cpu')
            # checkpoint_state_dict = checkpoint.pop('model')
            checkpoint_state_dict = checkpoint

            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        # print(k)
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    # print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

    return model
    
if __name__ == '__main__':
    model = YoloBody(num_classes=24,phi='s')
    inputs = torch.randn(1,3,224,224)
    outputs = model(inputs)
    
    # print(outputs[0].shape)
    
    for reg_feat, cls_feat in zip(outputs[0], outputs[1]):
        print(reg_feat.shape, cls_feat.shape)
        
'''
torch.Size([1, 64, 28, 28]) torch.Size([1, 64, 28, 28])
torch.Size([1, 64, 14, 14]) torch.Size([1, 64, 14, 14])
torch.Size([1, 64, 7, 7]) torch.Size([1, 64, 7, 7])
'''
    
    
    # model, fpn_dim = build_yolo_free(model_name='yolo_free_tiny', pretrained=False)
    # model.eval()
    
    # print(fpn_dim)

    # x = torch.randn(1, 3, 224, 224)
    # cls_feats, reg_feats = model(x)

    # for cls_feat, reg_feat in zip(cls_feats, reg_feats):
    #     print(cls_feat.shape, reg_feat.shape)
