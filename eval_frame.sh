# Frame mAP
python eval.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_tiny \
        --root /root/autodl-tmp/ \
        -bs 16 \
        -size 224 \
        -K 16 \
        --conf_thresh 0.005 \
        --weight weights/ucf24/yowo_v2_yolov8_3dshufflenetv2/yowo_v2_tiny_epoch_6.pth \
        --cal_frame_mAP \
        