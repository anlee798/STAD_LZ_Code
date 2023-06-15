# Frame mAP
python eval.py \
        --cuda \
        -d jhmdb21 \
        -v yowo_v2_tiny \
        --root /root/autodl-tmp/ \
        -bs 16 \
        -size 224 \
        -K 16 \
        --conf_thresh 0.005 \
        --weight weights/yowo_v2_tiny_epoch_20.pth \
        --cal_frame_mAP \
        