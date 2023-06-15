from models.yowo.yowov2_tsm_yolov7 import YOWO
from dataset.my_ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.transforms import BaseTransform
import torch
from utils.misc import MyCollateFunc,load_weight
import os
from tsm_yolov7 import DecodeBox
import numpy as np
from tqdm import tqdm

data_root = '/root/autodl-tmp/'
dataset = 'ucf24'
img_size = 224

# transform
basetransform = BaseTransform(img_size=224)

len_clip = 16

data_root = os.path.join(data_root, 'ucf24')

testset = UCF_JHMDB_Dataset(
                data_root=data_root,
                dataset=dataset,
                img_size=img_size,
                transform=basetransform,
                is_train=False,
                len_clip=len_clip,
                sampling_rate=1)

batch_size = 24

testloader = torch.utils.data.DataLoader(
            dataset=testset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=MyCollateFunc(), 
            num_workers=0,
            drop_last=False,
            pin_memory=True
            )

# cuda
device = torch.device("cuda")

# model
model = YOWO()
# load trained weight
model = load_weight(model=model, path_to_ckpt='weights/yowo_v2_TSM_epoch_7.pth')
model = model.to(device)

#
#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

anchors_path = 'config/yolo_anchors.txt'
classes_path = 'config/ucf_class.txt'
anchors, num_anchors      = get_anchors(anchors_path)
class_names, num_classes  = get_classes(classes_path)

num_classes = 24
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
bbox_util = DecodeBox(anchors, num_classes, (224, 224), anchors_mask)

input_shape = [224,224]
image_shape = [320,240]
letterbox_image = False

confidence = 0.001
nms_iou = 0.5


def judgeOutput(results,map_out_path,frame_ids):
    f = open(os.path.join(map_out_path, "detection-results/"+frame_ids), "w", encoding='utf-8') 
    if results[0] is None:
        return
    top_label   = np.array(results[0][:, 6], dtype = 'int32')
    top_conf    = results[0][:, 4] * results[0][:, 5]
    top_boxes   = results[0][:, :4]
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        # print("frame_ids",frame_ids,",predicted_class",predicted_class)
        box             = top_boxes[i]
        score           = str(top_conf[i])

        top, left, bottom, right = box
        if predicted_class not in class_names:
            continue
        f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
    f.close()

map_out_path = 'map_out'
if not os.path.exists(map_out_path):
    os.makedirs(map_out_path)
if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
    os.makedirs(os.path.join(map_out_path, 'detection-results'))

# inference
for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(tqdm(testloader)):
        
    # to device
    batch_video_clip = batch_video_clip.to(device)
    with torch.no_grad():
        # inference
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        outputs = model(batch_video_clip)
        
        # process batch
        for bi, (output1, output2, output3) in enumerate(zip(outputs[0], outputs[1], outputs[2])):
            frame_id = batch_frame_id[bi]
            video_clip = batch_video_clip[bi]
            output = []
            output1 = output1.unsqueeze(0)
            output.append(output1)
            output2 = output2.unsqueeze(0)
            output.append(output2)
            output3 = output3.unsqueeze(0)
            output.append(output3)
            
            outputs = bbox_util.decode_box(output)
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape, 
                            image_shape, letterbox_image, conf_thres = confidence, nms_thres = nms_iou)
            
            judgeOutput(results,map_out_path,frame_id)
                                                   