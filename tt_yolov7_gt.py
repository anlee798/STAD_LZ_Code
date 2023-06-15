import os
from tqdm import tqdm
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
classes_path = 'config/ucf_class.txt'
class_names, num_classes  = get_classes(classes_path)

map_out_path = 'map_out'
if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
    os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    
folder_path = '/root/autodl-tmp/YOWOv2_TSM_yolov7/evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/'

for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith('.txt'):
        # print("filename",filename)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            with open(os.path.join(map_out_path, "ground-truth/"+filename), "w") as new_f:
                for line in file:
                    line = line.strip()
                    if line:
                        # print("filename",filename,"line",line)
                        # 在这里处理非空行
                        # print(class_names[int(line.split(' ')[0])-1])
                        #line = line.replace(line.split(' ')[0], class_names[int(line.split(' ')[0])-1])
                        # line = class_names[int(line.split(' ')[0])-1] +' '+ \
                        #     line.split(' ')[1] +' '+ \
                        #     line.split(' ')[2] +' '+ \
                        #     line.split(' ')[3] +' '+ \
                        #     line.split(' ')[4] +' '
                        # print(line)
                        new_f.write(line)
# with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:

# folder_path = '/root/autodl-tmp/YOWOv2_TSM_yolov7/map_out/detection-results/'
# number = 0
# for filename in tqdm(os.listdir(folder_path)):
#     if filename.endswith('.txt'):
#         number += 1
        
# print(number)
#137557