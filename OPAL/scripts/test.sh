
# Cityscapes

CUDA_VISIBLE_DEVICES=0,1,2,3 python osprey/eval/eval_open_vocab_seg_full_metric_set.py \
    --dataset cityscapes --model pretained_weights/pretrained_weights/Osprey-7b \
    --bert 'sentence-transformers/all-MiniLM-L6-v2' --num-gpus 1 \

# ADE20K

CUDA_VISIBLE_DEVICES=0,1,2,3 python osprey/eval/eval_open_vocab_seg_detectron2.py \
    --dataset ade --model pretained_weights/stage3+contrastive_lr1e-5_bs16_wu10_cg1_all_data_ep4 \
    --bert 'sentence-transformers/all-MiniLM-L6-v2' --num-gpus 1 --seed 1 \

