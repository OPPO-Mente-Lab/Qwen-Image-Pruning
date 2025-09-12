import torch
import os
from diffusers import FlowMatchEulerDiscreteScheduler, DiffusionPipeline

if __name__ == "__main__":
    model_name = "/mnt/workspace/group/models/Qwen-Image-Pruning"
    lora_name = '/mnt/workspace/group/models/qwen-image-realism-lora'

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.bfloat16
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe.load_lora_weights(lora_name, adapter_name="lora")
    pipe = pipe.to(device)

    # Generate image
    positive_magic = {"en": ", Ultra HD, 4K, cinematic composition.", # for english prompt,
    "zh": "，超清，4K，电影级构图。" # for chinese prompt,
    }
    negative_prompt = " "

    prompts = [
        '一个穿着"QWEN"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 "一、Qwen-Image的技术路线： 探索视觉生成基础模型的极限，开创理解与生成一体化的未来。二、Qwen-Image的模型特色：1、复杂文字渲染。支持中英渲染、自动布局； 2、精准图像编辑。支持文字编辑、物体增减、风格变换。三、Qwen-Image的未来愿景：赋能专业内容创作、助力生成式AI发展。"',
        '海报，温馨家庭场景，柔和阳光洒在野餐布上，色彩温暖明亮，主色调为浅黄、米白与淡绿，点缀着鲜艳的水果和野花，营造轻松愉快的氛围，画面简洁而富有层次，充满生活气息，传达家庭团聚与自然和谐的主题。文字内容：“共享阳光，共享爱。全家一起野餐，享受美好时光。让每一刻都充满欢笑与温暖。”',
        '一个穿着校服的年轻女孩站在教室里，在黑板上写字。黑板中央用整洁的白粉笔写着“Introducing Qwen-Image, a foundational image generation model that excels in complex text rendering and precise image editing”。柔和的自然光线透过窗户，投下温柔的阴影。场景以写实的摄影风格呈现，细节精细，景深浅，色调温暖。女孩专注的表情和空气中的粉笔灰增添了动感。背景元素包括课桌和教育海报，略微模糊以突出中心动作。超精细32K分辨率，单反质量，柔和的散景效果，纪录片式的构图。',
        '一个台球桌上放着两排台球，每排5个，第一行的台球上面分别写着"Qwen""Image" "将 "于" "8" ，第二排台球上面分别写着"月" "正" "式" "发" "布" 。',
    ]
    
    output_dir = 'examples_Pruning+Realism_LoRA'
    os.makedirs(output_dir, exist_ok=True)
    for prompt in prompts:
        output_img_path = f"{output_dir}/{prompt[:80]}.png"
        image = pipe(
            prompt=prompt + positive_magic['zh'],
            negative_prompt=negative_prompt,
            width=1328,
            height=1328,
            num_inference_steps=8,
            true_cfg_scale=1,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]
        image.save(output_img_path)