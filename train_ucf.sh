# Train YOWOv2 on UCF24 dataset
python train.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_tiny \
        -mname YOWOv3 \
        --root /root/autodl-tmp/ \
        --num_workers 10 \
        --eval_epoch 1 \
        --max_epoch 7 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 80 \
        -accu 16 \
        -K 16 \
        --loss_conf_weight 1 \
        --loss_cls_weight 1 \
        --loss_reg_weight 5 \
        # --eval \
