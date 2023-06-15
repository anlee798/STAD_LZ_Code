import torch.nn as nn
from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual

base_model = mobilenet_v2(True)

base_model.last_layer_name = 'classifier'
input_size = 224
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

is_shift  =True

base_model.avgpool = nn.AdaptiveAvgPool2d(1)
if is_shift:
    from ops.temporal_shift import TemporalShift
    for m in base_model.modules():
        # print("m:",m)
        if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
            print("m:",m)
            m.conv[0] = TemporalShift(m.conv[0], n_segment=8, n_div=8)
            
'''
 python ops/tt.py
'''