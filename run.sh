#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_small --data_path /var/rosbags/vid/hella/frames --output_dir /var/rosbags/vid/hella_opt3 --epochs 300 --freeze_last_layer 5 --use_fp16 False --batch_size_per_gpu 8 --patch_size 8

python video_generation.py --input_path /var/rosbags/vid/20210423-140241+0200--wald-ollenhauer_sensors_hella_image.mp4 --output_path /var/rosbags/vid/hella_trained_opt3 --pretrained_weights /var/rosbags/vid/hella_trained_opt3/checkpoint.pth --fps 30 --patch_size 8 --no_extract
