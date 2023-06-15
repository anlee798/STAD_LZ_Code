import torch
import torch.nn as nn

try:
    from .backbone import Backbone, Multi_Concat_Block, Conv
except:
    from backbone import Backbone, Multi_Concat_Block, Conv

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.cv3(torch.cat([m(x1) for m in self.m] + [x1], 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

# My YOLOv7 Head
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


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes,phi='s', pretrained=False):
        super(YoloBody, self).__init__()
        #-----------------------------------------------#
        #   定义了不同yolov7-tiny的参数
        #-----------------------------------------------#
        transition_channels = 16
        block_channels      = 16
        panet_channels      = 16
        e                   = 1
        n                   = 2
        ids                 = [-1, -2, -3, -4]
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        #---------------------------------------------------#
        self.backbone   = Backbone(transition_channels, block_channels, n, pretrained=pretrained)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.sppcspc                = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2         = Conv(transition_channels * 16, transition_channels * 8)
        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1         = Conv(transition_channels * 8, transition_channels * 4)
        self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1           = Conv(transition_channels * 4, transition_channels * 8, k=3, s=2)
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2           = Conv(transition_channels * 8, transition_channels * 16, k=3, s=2)
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)

        self.rep_conv_1 = Conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = Conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = Conv(transition_channels * 16, transition_channels * 32, 3, 1)

        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)
        
        
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 
        self.head       = YOLOXHead2(num_classes, width, depthwise=depthwise)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        
        P5          = self.sppcspc(feat3)
        P5_conv     = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4          = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4          = self.conv3_for_upsample1(P4)

        P4_conv     = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3          = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3          = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)
        
        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)
        # #---------------------------------------------------#
        # #   第三个特征层
        # #   y3=(batch_size, 75, 80, 80)
        # #---------------------------------------------------#
        # out2 = self.yolo_head_P3(P3)
        # #---------------------------------------------------#
        # #   第二个特征层
        # #   y2=(batch_size, 75, 40, 40)
        # #---------------------------------------------------#
        # out1 = self.yolo_head_P4(P4)
        # #---------------------------------------------------#
        # #   第一个特征层
        # #   y1=(batch_size, 75, 20, 20)
        # #---------------------------------------------------#
        # out0 = self.yolo_head_P5(P5)

        # return [out0, out1, out2]
        # fpn_outs = (P5,P4,P3)
        fpn_outs = (P3,P4,P5)
        '''
        P3 torch.Size([1, 128, 28, 28])
        P4 torch.Size([1, 256, 14, 14])
        P5 torch.Size([1, 512, 7, 7])
        '''
        outputs1,outputs2     = self.head.forward(fpn_outs)
        return outputs1,outputs2

if __name__ == '__main__':
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model = YoloBody(anchors_mask=anchors_mask,num_classes=24)
    inputs = torch.randn(1,3,224,224)
    outputs = model(inputs)
    # print(outputs[0].size())
    # print(outputs[1].size())
    # print(outputs[2].size())
    for reg_feat, cls_feat in zip(outputs[0], outputs[0]):
        print(reg_feat.shape, cls_feat.shape)
