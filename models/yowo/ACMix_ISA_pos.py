import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
import time

# def position(H, W, is_cuda=True):
#     if is_cuda:
#         loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1)
#         loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0) 
#     else:
#         loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1)
#         loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0)

#     loc_h = loc_h.repeat(1, W)
#     loc_w = loc_w.repeat(H, 1)
    
#     loc = torch.cat([loc_h.unsqueeze(0), loc_w.unsqueeze(0)], 0).unsqueeze(0)
#     return loc

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

class SelfAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(self.key_channels, self.key_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(self.key_channels, self.key_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
        )

        self.f_value = nn.Conv2d(
            self.in_channels, self.value_channels, kernel_size=1, bias=False)
        self.W = nn.Sequential(
            nn.Conv2d(self.value_channels, self.out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x, feat_q, feat_k, feat_v):
        if x == None:
            batch_size, h, w = feat_q.size(0), feat_q.size(2), feat_q.size(3)

            value = feat_v.view(batch_size, self.value_channels, -1)
            value = value.permute(0, 2, 1)
            query = feat_q.view(batch_size, self.key_channels, -1)
            query = query.permute(0, 2, 1)
            key = feat_k.view(batch_size, self.key_channels, -1)
        else:
            batch_size, h, w = x.size(0), x.size(2), x.size(3)

            value = self.f_value(x).view(batch_size, self.value_channels, -1)
            value = value.permute(0, 2, 1)
            query = self.f_query(x).view(batch_size, self.key_channels, -1)
            query = query.permute(0, 2, 1)
            key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.W(context)
        return context


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1, down_factor=[8, 8]):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.down_factor = down_factor
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3*self.head, self.kernel_conv *
                            self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)

        self.long_range_sa = SelfAttentionBlock2D(
            in_channels=out_planes, key_channels=out_planes, value_channels=out_planes, out_channels=out_planes, bn_type=None)
        self.short_range_sa = SelfAttentionBlock2D(
            in_channels=out_planes, key_channels=out_planes, value_channels=out_planes, out_channels=out_planes, bn_type=None)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv,
                             self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i//self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w)

        # ISA Block
        dh, dw = self.down_factor       # down_factor for h and w, respectively
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        
        # pad the feature if the size is not divisible
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:  # padding in both left&right sides
            feats_q = F.pad(q_att, (pad_w//2, pad_w - pad_w // 2, pad_h//2, pad_h - pad_h//2))
            feats_k = F.pad(k_att, (pad_w//2, pad_w - pad_w // 2, pad_h//2, pad_h - pad_h//2))
            feats_v = F.pad(v_att, (pad_w//2, pad_w - pad_w // 2, pad_h//2, pad_h - pad_h//2))
        else:
            feats_q = q_att
            feats_k = k_att
            feats_v = v_att
            
        # 加入位置编码
        _,_,pos_h,pos_w = feats_q.shape
        pos = position(pos_h, pos_w, x.is_cuda)
        pos = self.conv_p(pos)
        feats_q = feats_q + pos
        feats_k = feats_k + pos

        # long range attention
        feats_q = feats_q.view(b, c, out_h, dh, out_w, dw)
        feats_k = feats_k.view(b, c, out_h, dh, out_w, dw)
        feats_v = feats_v.view(b, c, out_h, dh, out_w, dw)
        feats_q = feats_q.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)
        feats_k = feats_k.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)
        feats_v = feats_v.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)
        feats = self.long_range_sa(None, feats_q, feats_k, feats_v)
        c = self.out_planes

        # short range attention
        feats = feats.view(b, dh, dw, c, out_h, out_w)
        feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw)
        feats = self.short_range_sa(feats, None, None, None)
        feats = feats.view(b, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = feats.contiguous().view(b, c, dh * out_h, dw * out_w)

        # remove padding
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]

        out_att = feats

# # ### att
#         # ## positional encoding
#         pe = self.conv_p(position(h, w, x.is_cuda))

#         q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
#         k_att = k.view(b*self.head, self.head_dim, h, w)
#         v_att = v.view(b*self.head, self.head_dim, h, w)

#         if self.stride > 1:
#             q_att = stride(q_att, self.stride)
#             q_pe = stride(pe, self.stride)
#         else:
#             q_pe = pe

#         unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # b*head, head_dim, k_att^2, h_out, w_out
#         unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # 1, head_dim, k_att^2, h_out, w_out

#         att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
#         att = self.softmax(att)

#         out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
#         out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

# conv
        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h*w), k.view(
            b, self.head, self.head_dim, h*w), v.view(b, self.head, self.head_dim, h*w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(
            x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


# start_time = time.time()
# models = ACmix(3, 16, head=4)

# inputs = torch.randn(1, 3, 224, 224)

# outputs = models(inputs)

# print(outputs.shape)
# end_time = time.time()
# print("cost time", end_time-start_time)
