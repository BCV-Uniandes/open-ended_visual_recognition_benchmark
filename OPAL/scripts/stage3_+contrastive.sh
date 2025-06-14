#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH

deepspeed --include localhost:0,1,2,3 osprey/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --dataset_config ./osprey/configs/stage3.json \
    --version v1 \
    --model_name_or_path /media/nvme0/experiments/stage2+contrastive_lr5e-5_bs32_wu10_cg1_all_data_ep4 \
    --vision_tower  pretrained_weights/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir './exps/stage3' \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "none" \
    --group_by_modality_length False \


    #   --model_name_or_path /media/nvme0/experiments/stage2+contrastive_lr5e-5_bs32_wu10_cg1_all_data_ep4 \
