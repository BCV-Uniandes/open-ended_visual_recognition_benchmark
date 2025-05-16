CUDA_VISIBLE_DEVICES=0 python -m LAC.eval_open_vocab_seg_full_metric_open \
    --dataset ade \
    --model opal \
    --model_outputs_path outputs/opal/ade20k/descriptions.json \
    --semantic_relationship_file_path ./LAC//semantic_relationships/output_semantics_gpt4_ade20k.json \
    --num-gpus 1 \

