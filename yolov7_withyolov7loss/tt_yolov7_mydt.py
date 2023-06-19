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
if not os.path.exists(os.path.join(map_out_path, 'mydetetction-truth')):
    os.makedirs(os.path.join(map_out_path, 'mydetetction-truth'))
    
folder_path = '/root/autodl-tmp/YOWOv2_TSM_yolov7/map_out/detection-results/'

number = 0
for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith('.txt'):
        number += 1
        # print("filename",filename)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            with open(os.path.join(map_out_path, "mydetetction-truth/"+filename), "w") as new_f:
                for line in file:
                    line = line.strip()
                    if line:
                        # print(line)
                        cur_index = 0
                        for index, class_name in enumerate(class_names):
                            if line.split(' ')[0] == class_name:
                                cur_index = index
                                continue
                        line = line.replace(line.split(' ')[0], str(cur_index))
                        new_f.write(line+'\n')

# folder_path = '/root/autodl-tmp/YOWOv2_TSM_yolov7/map_out/mydetetction-truth/'
# number = 0
# for filename in tqdm(os.listdir(folder_path)):
#     if filename.endswith('.txt'):
#         number += 1
#     else:
#         print(filename)
        
print(number)
#137557