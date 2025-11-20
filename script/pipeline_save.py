import argparse
import math
import os
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import gc
import json

from src.transformer_qwenimage_no_txt_no_ffn import QwenImageTransformer2DModel as QITf2DM_no_txt_no_ffn
from src.transformer_qwenimage_pruning import QwenImageTransformer2DModel as QITF2DM
from diffusers import QwenImagePipeline, DiffusionPipeline

def save_pipeline(pipe, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 3. 保存整个 pipeline
    # 这会将模型、分词器和配置文件都保存到指定目录
    print(f"正在将 pipeline 保存到 '{save_directory}'...")
    pipe.save_pretrained(save_directory)

    print("保存完成！")

    # 查看保存了哪些文件
    print("\n目录中的文件：")
    for filename in os.listdir(save_directory):
        print(f"- {filename}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./output")
    parser.add_argument("--layer_indexs", type=str, default='[[3,4],[5,7],[8,10],[11,12],[15,24],[25,27],[29,30],[42,43],[45,47],[48,49],[52,53],[54,55],[56,57]]')
    parser.add_argument('--ehs_same', type=str,default='[[13,14],[31,32],[33,34],[36,37],[40,41],[50,59]]')
    parser.add_argument('--ehs_same_indexs', type=int, nargs='*', default=[3, 4, 5])
    parser.add_argument('--ffn_block_indexs',type=int, nargs='*', default=[2, 58, 59])
    parser.add_argument('--removed_block', type=int, nargs='*',default=[])
    parser.add_argument("--use_ehs_proj", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-Image")
    parser.add_argument("--save_dir", type=str, default="./output")
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.checkpoint_dir, 'diffusion_pytorch_model.bin')
    layer_indexs = json.loads(args.layer_indexs)
    ehs_same = json.loads(args.ehs_same)
    ehs_same_indexs = args.ehs_same_indexs
    ffn_block_indexs = args.ffn_block_indexs
    removed_block = args.removed_block
    use_ehs_proj = bool(args.use_ehs_proj)

    model_name = args.model_name
    save_dir = args.save_dir

    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.bfloat16
        device = "cpu"

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    # 模型初始化
    model = QITf2DM_no_txt_no_ffn.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    ).to(device)
    model.init_distill(layer_indexs=layer_indexs, ehs_same=ehs_same)

    # 导入训练的权重
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    transformer_distill_blocks_state_dict = state_dict["transformer_distill_blocks"]
    attn_proj_list_state_dict = state_dict["attn_proj_list"]
    ffn_proj_list_state_dict = state_dict["ffn_proj_list"]

    model.transformer_distill_blocks.load_state_dict(transformer_distill_blocks_state_dict)
    model.attn_proj_list.load_state_dict(attn_proj_list_state_dict)
    model.ffn_proj_list.load_state_dict(ffn_proj_list_state_dict)

    if "ehs_proj_list" in state_dict:
        ehs_proj_list_state_dict = state_dict["ehs_proj_list"]
        model.ehs_proj_list.load_state_dict(ehs_proj_list_state_dict)

    model.to(dtype=torch.bfloat16)

    # 创建文件路径
    os.makedirs(save_dir, exist_ok=True)
    if not removed_block:
        checkpoint_path = os.path.join(save_dir, f'{args.checkpoint_dir.split("/")[-2]}_{args.checkpoint_dir.split("/")[-1]}_{ehs_same}_{ehs_same_indexs}_{ffn_block_indexs}.bin')
    else:
        checkpoint_path = os.path.join(save_dir, f'{args.checkpoint_dir.split("/")[-2]}_{args.checkpoint_dir.split("/")[-1]}_{ehs_same}_{ehs_same_indexs}_{ffn_block_indexs}_{removed_block}.bin')
    if use_ehs_proj:
        checkpoint_path = checkpoint_path.replace('.bin', f'_use_ehs_proj.bin')
    
    # 合并权重，并输出模型参数
    single_stream_index, ffn_proj_index, ehs_proj_index, num_layers = model.save_distill(ehs_same, ehs_same_indexs, ffn_block_indexs, use_ehs_proj, checkpoint_path, removed_block)
    print(single_stream_index, ffn_proj_index, ehs_proj_index, num_layers)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    


    # 获取文件路径
    os.makedirs(save_dir, exist_ok=True)
    if not removed_block:
        checkpoint_path = os.path.join(save_dir, f'{args.checkpoint_dir.split("/")[-2]}_{args.checkpoint_dir.split("/")[-1]}_{ehs_same}_{ehs_same_indexs}_{ffn_block_indexs}.bin')
    else:
        checkpoint_path = os.path.join(save_dir, f'{args.checkpoint_dir.split("/")[-2]}_{args.checkpoint_dir.split("/")[-1]}_{ehs_same}_{ehs_same_indexs}_{ffn_block_indexs}_{removed_block}.bin')
    if use_ehs_proj:
        checkpoint_path = checkpoint_path.replace('.bin', f'_use_ehs_proj.bin')
    
    # 用刚刚输出的模型参数初始化一个新模型
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model = QITF2DM(
        num_layers=num_layers, single_stream_index=single_stream_index, 
        ffn_proj_index=ffn_proj_index, ehs_proj_index=ehs_proj_index
    )
    model.load_state_dict(state_dict)
    model.to(dtype=torch.bfloat16)

    # 初始化一个pipeline

    pipe = QwenImagePipeline.from_pretrained(
        model_name, transformer=model, torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)

    # 将pipeline保存成文件夹
    if not removed_block:
        save_directory = f"{save_dir}/Qwen-Image-Pruning"
    else:
        save_directory = f"{save_dir}/Qwen-Image-Pruning_{removed_block}"

    save_pipeline(pipe, save_directory)
