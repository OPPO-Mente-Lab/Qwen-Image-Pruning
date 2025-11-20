export CC=gcc
export CXX=g++
# WORKING_DIR="qwen-image-pd"
# cd $WORKING_DIR
# if [[ ":$PYTHONPATH:" != *":$WORKING_DIR:"* ]]; then export PYTHONPATH="$WORKING_DIR:$PYTHONPATH"; fi

accelerate launch --config_file "accelerate_config.yaml" \
    --machine_rank 0 \
    --main_process_ip "127.0.0.1" \
    --main_process_port 29001 \
    --num_machines 1 \
    --num_processes 1 \
    script/train_phase.py \
    --webdataset_base_urls "/mnt/workspace/group/text2img_data/text_rendering_space_qwen_image/*/*" \
    --num_workers 1 \
    --batch_size 1 \
    --shard_width 5 \
    --train_split 1.0 \
    --val_split 0.0 \
    --test_split 0.0 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 20000 \
    --learning_rate 1e-05 \
    --lr_scheduler "cosine" \
    --lr_warmup_steps 16 \
    --mixed_precision "bf16" \
    --checkpointing_steps 1000 \
    --output_dir "output/train_phase" \
    --max_grad_norm 1e+30 \
    --checkpoints_total_limit 5 \
    --use_8bit_adam \
    --teacher "/mnt/workspace/group/models/Qwen-Image/Qwen-Image" \
    --layer_index "[[3,4],[5,7],[8,10],[11,12],[15,24],[25,27],[29,30],[42,43],[45,47],[48,49],[52,53],[54,55],[56,57]]" \
