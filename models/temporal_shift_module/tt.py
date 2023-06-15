import torch
from opts import parser
from ops.models import TSN
from thop import profile

# args = parser.parse_args()
num_class = 24

model = TSN(num_class, num_segments=16, modality='RGB',
                base_model='mobilenetv2',
               #  consensus_type=args.consensus_type,
                dropout=0.5,
                img_feature_dim=224,
                partial_bn=True,
               #  pretrain=args.pretrain,
                is_shift=True, shift_div=8, shift_place="blockres",
                fc_lr5=True,
               #  temporal_pool=args.temporal_pool,
                non_local=False)

inputs = torch.randn(1,3,16,224,224)

outputs = model(inputs)

print(outputs.size())

flops, params = profile(model, inputs=(inputs,))
total = sum([param.nelement() for param in model.parameters()])
print('   Number of params: %.2fM' % (total / 1e6))
print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))

'''
python tt.py ucf101 RGB --arch mobilenetv2 --num_segments 8 --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres

python tt.py ucf101 RGB \--arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres
'''

