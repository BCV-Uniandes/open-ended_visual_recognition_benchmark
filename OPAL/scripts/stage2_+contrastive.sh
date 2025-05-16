#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH
export TRAIN_MASK_MODULE=1

deepspeed --include localhost:0,1,2,3 osprey/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path  pretrained_weights/vicuna-7b-v1.5 \
    --dataset_config ./osprey/configs/stage2.json \
    --version v1 \
    --vision_tower  pretrained_weights/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin \
    --pretrain_mm_mlp_adapter pretrained_weights/osprey-v1.0-mlp2x-512px-convnext-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir './exps/stage2+contrastive_lr5e-5_bs32_wu10_cg1_all_data_ep4_debug' \
    --num_train_epochs 4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2961 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
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
