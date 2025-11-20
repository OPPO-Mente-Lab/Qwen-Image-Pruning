# WORKING_DIR="/mnt/workspace/group/***/emotion/qwen-image-pd"
# cd $WORKING_DIR
# if [[ ":$PYTHONPATH:" != *":$WORKING_DIR:"* ]]; then export PYTHONPATH="$WORKING_DIR:$PYTHONPATH"; fi

python script/inference_txt_ffn.py \
    --width 1328 \
    --height 1328 \
    --model_name "output/pipeline_save/Qwen-Image-Pruning" \
    --save_path "output/qwen-image-pruning" \