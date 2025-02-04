#!/bin/bash

# output_correlation: enc_feat, dec_feat, ca_map 

CUDA=4
CUDA_VISIBLE_DEVICES=${CUDA} python -u eval_matching.py \
    --seed 1997 \
    --dataset hp \
    --eval_img_size 240 240 \
    --model_img_size 224 224 \
    --dense_zoom_in \
    --dense_zoom_ratio 2 3 \
    --model crocov2 \
    --croco_ckpt ./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 1e-4 \
    --save_dir ./vis/eval/hp240/zoom34/ZeroCo_LargeBase \
    --log_warped_images \


