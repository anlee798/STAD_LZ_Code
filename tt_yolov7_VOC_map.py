from utils_map import get_map
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')
#--------------------------------------------------------------------------------------#
#   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
#   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
#
#   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
#   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
#--------------------------------------------------------------------------------------#
MINOVERLAP      = 0.5
#---------------------------------------------------------------------------------------------------------------#
#   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
#   
#   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
#   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
#   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
#---------------------------------------------------------------------------------------------------------------#
score_threhold  = 0.5
map_out_path = 'map_out'
get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)