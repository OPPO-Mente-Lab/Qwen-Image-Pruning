# WORKING_DIR="/mnt/workspace/group/***/emotion/qwen-image-pd"
# cd $WORKING_DIR
# if [[ ":$PYTHONPATH:" != *":$WORKING_DIR:"* ]]; then export PYTHONPATH="$WORKING_DIR:$PYTHONPATH"; fi

python script/pipeline_save.py \
    --checkpoint_dir "output/train_phase_ehs_ffn_proj_initlinear/10" \
    --model_name "/mnt/workspace/group/models/Qwen-Image/Qwen-Image" \
    --save_dir "output/pipeline_save" \
    --layer_indexs '[[3,4],[5,7],[8,10],[11,12],[15,24],[25,27],[29,30],[42,43],[45,47],[48,49],[52,53],[54,55],[56,57]]' \
    --ehs_same '[[13,14],[31,32],[33,34],[36,37],[40,41],[50,59]]' \
    --ehs_same_indexs 3 4 5 \
    --ffn_block_indexs 2 58 59 \
    --removed_block \
    --use_ehs_proj 1 \