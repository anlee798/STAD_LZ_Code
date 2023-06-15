import argparse
import torch
import os

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator

from dataset.transforms import BaseTransform

from utils.misc import load_weight, CollateFunc

from config import build_dataset_config, build_model_config
from models import build_model

from utils.misc import MyCollateFunc, build_dataset, build_dataloader,CollateFunc
from tsm_yolov7 import TSM_YOLOV7
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')

    # basic
    parser.add_argument('-bs', '--batch_size', default=1, type=int,
                        help='test batch size')
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_path', default='evaluator/eval_results/',
                        type=str, help='Trained state_dict file path to open')

    # dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb, ava_v2.2.')
    parser.add_argument('--root', default='/root/autodl-tmp/',
                        help='data root')

    # eval
    parser.add_argument('--cal_frame_mAP', action='store_true', default=False, 
                        help='calculate frame mAP.')
    parser.add_argument('--cal_video_mAP', action='store_true', default=False, 
                        help='calculate video mAP.')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold. We suggest 0.5 for UCF24 and AVA.')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk prediction candidates.')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")
    parser.add_argument('--freeze_backbone_2d', action="store_true", default=False,
                        help="freeze 2D backbone.")
    parser.add_argument('--freeze_backbone_3d', action="store_true", default=False,
                        help="freeze 3d backbone.")
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
    
    # Batchsize
    parser.add_argument('-tbs', '--test_batch_size', default=12, type=int, 
                        help='test batch size on a single GPU.')
    parser.add_argument('--eval', action='store_true', default=False, 
                        help='do evaluation during training.')
    parser.add_argument('--num_workers', default=0, type=int, 
                        help='Number of workers used in dataloading')
    
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

def ucf_jhmdb_eval(args, d_cfg, model, transform, collate_fn):
    data_dir = os.path.join(args.root, 'ucf24')
    if args.cal_frame_mAP:
        # Frame mAP evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=args.img_size,
            len_clip=args.len_clip,
            batch_size=args.batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            save_path=args.save_path
            )
        # evaluate
        evaluator.evaluate_frame_map(model, show_pr_curve=True)
        
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
        
if __name__ == '__main__':
    args = parse_args()
    num_classes = 24
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)
    
    # # build model
    # model, _ = build_model(
    #     args=args, 
    #     d_cfg=d_cfg,
    #     m_cfg=m_cfg,
    #     device=device, 
    #     num_classes=num_classes, 
    #     trainable=True
    #     )
    # # load trained weight
    # model = load_weight(model=model, path_to_ckpt=args.weight)

    # # to eval
    # model = model.to(device).eval()

    # transform
    basetransform = BaseTransform(img_size=args.img_size)
    
    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=False)
    
    batch_size = 4
    
    dataloader = build_dataloader(args, dataset, batch_size, MyCollateFunc(), is_train=False)
    
    print("Load model.")
    model = TSM_YOLOV7()
    map_out_path = 'map_out'
    class_names, _ = get_classes('config/ucf_class.txt')
    
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    
    for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
        # print("iter_i",iter_i)
        print("frame_ids",frame_ids)
        # print("video_clips",video_clips.shape)
        # print("targets",targets[0].shape)
        # print("targets",targets[0])
        image_data = video_clips[:, :, -1, :, :]
        # print(str(iter_i))
        #model.get_map_txt(str(frame_ids).split('.')[0],video_clips,image_data,map_out_path,class_names)
        import sys
        sys.exit(0)
    