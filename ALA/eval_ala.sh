CUDA_VISIBLE_DEVICES=0 python -m ALA.eval_open_vocab_seg_full_metric_open \
    --dataset ade \
    --model maskclip \
    --model_outputs_path /home/cigonzalez/code/open-ended_visual_recognition_benchmark/outputs/maskclip/ade20k/descriptions.json \
    --semantic_relationship_file_path ./ALA//semantic_relationships/output_semantics_gpt4_ade20k.json \
    --num-gpus 1 \

