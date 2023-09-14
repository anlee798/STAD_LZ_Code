import argparse
from models import build_model
from config import build_dataset_config, build_model_config
import torch
from torch.autograd import Variable
from utils.misc import CollateFunc, build_dataset, build_dataloader
from thop import profile
import warnings
from utils.misc import load_weight
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # Visualization
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='./weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='use tensorboard')

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False, 
                        help='do evaluation during training.')
    parser.add_argument('--eval_epoch', default=1, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--save_dir', default='inference_results/',
                        type=str, help='save inference results.')
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')

    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on a single GPU.')
    parser.add_argument('-tbs', '--test_batch_size', default=16, type=int, 
                        help='test batch size on a single GPU.')
    parser.add_argument('-accu', '--accumulate', default=1, type=int, 
                        help='gradient accumulate.')
    parser.add_argument('-lr', '--base_lr', default=0.0001, type=float, 
                        help='base lr.')
    parser.add_argument('-ldr', '--lr_decay_ratio', default=0.5, type=float, 
                        help='base lr.')

    # Epoch
    parser.add_argument('--max_epoch', default=10, type=int, 
                        help='max epoch.')
    parser.add_argument('--lr_epoch', nargs='+', default=[2,3,4], type=int,
                        help='lr epoch to decay')

    # Model
    parser.add_argument('-v', '--version', default='yowo_v2_tiny', type=str, #yowo_v2_tiny
                        help='build YOWOv2')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold. We suggest 0.5 for UCF24 and AVA.')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk prediction candidates.')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('--freeze_backbone_2d', action="store_true", default=False,
                        help="freeze 2D backbone.")
    parser.add_argument('--freeze_backbone_3d', action="store_true", default=False,
                        help="freeze 3d backbone.")
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    # Dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, ava_v2.2')
    # parser.add_argument('--root', default='/mnt/share/ssd2/dataset/STAD/',/Volumes/Extreme/yowov2_data/
    #                     help='data root')  /Users/zhuanlei/Documents/WorkSpace/ActionRecognition/myucf24/
    # /Volumes/Extreme/yowov2_data/
    parser.add_argument('--root', default='/root/autodl-tmp/',
                        help='data root')
    parser.add_argument('--num_workers', default=0, type=int, 
                        help='Number of workers used in dataloading')

    # Matcher
    parser.add_argument('--center_sampling_radius', default=2.5, type=float, 
                        help='conf loss weight factor.')
    parser.add_argument('--topk_candicate', default=10, type=int, 
                        help='cls loss weight factor.')

    # Loss
    parser.add_argument('--loss_conf_weight', default=1, type=float, 
                        help='conf loss weight factor.')
    parser.add_argument('--loss_cls_weight', default=1, type=float, 
                        help='cls loss weight factor.')
    parser.add_argument('--loss_reg_weight', default=5, type=float, 
                        help='reg loss weight factor.')
    parser.add_argument('-fl', '--focal_loss', action="store_true", default=False,
                        help="use focal loss for classification.")
    
    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


args = parse_args()
# config
d_cfg = build_dataset_config(args)
m_cfg = build_model_config(args)

device = torch.device('cuda')

#Model
model, criterion = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=24, 
        trainable=True,
        resume=args.resume
        )

# YOWOv2
# model = load_weight(model,'weights/yowo_v2_tiny_epoch_6.pth')

# YOWOv3_TSM ACMIX
# model = load_weight(model,'weights/yowo_v3_TSM.pth')

# YOWOv3 ACISA
model = load_weight(model,'weights/yowo_v3_ACISA.pth')

# YOWOv3 
# model = load_weight(model,'weights/yowo_v3.pth')

# # YOWOv3 ACMIX
# model = load_weight(model,'weights/yowo_v3_withacmix.pth')

# # YOWOv3 Without ACMIX
# model = load_weight(model,'weights/yowo_v3_without_acimx.pth')

model = model.to(device)

model.eval()

input_batch = torch.randn(1,3,16,224,224).to(device)

# 关闭梯度计算
torch.set_grad_enabled(False)

# 执行推理
output = None
with torch.no_grad():
    # 在计时器开始之前执行一次推理以避免冷启动效应
    output = model(input_batch)

    # 开始计时
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    # 执行推理
    output = model(input_batch)

    # 结束计时
    end_time.record()
    torch.cuda.synchronize()  # 等待GPU操作完成

# 计算推理时间
inference_time = start_time.elapsed_time(end_time)  # 毫秒

print(f"Inference time on GPU: {inference_time} ms")
print(f"Inference time on GPU: {1000/inference_time} FPS")



# import time 
# alltime = 0
# allframe = 0

# with torch.no_grad():
#     for i in range(201):
#         start_time = time.time()
#         if i == 0:
#             continue
#         outputs = model(input_batch)
#         end_time = time.time()
#         alltime += (end_time - start_time)
#         allframe += 1
#         print("per time:",end_time - start_time)
    

# print("alltime: ", alltime/allframe)
# print("alltime: ", 1/(alltime/allframe))