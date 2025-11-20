import os
import json
import glob

import torch
from diffusers import QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler, DiffusionPipeline

from diffusers.utils import load_image
from diffusers import QwenImageControlNetPipeline, QwenImageControlNetModel

if __name__ == '__main__':
    model_name = "OPPOer/Qwen-Image-Pruning"
    controlnet_name = "InstantX/Qwen-Image-ControlNet-Union"

    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.bfloat16
        device = "cpu"

    controlnet = QwenImageControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.bfloat16)

    pipe = QwenImageControlNetPipeline.from_pretrained(
        model_name, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)

    # Generate image
    prompt_dict = {
        "soft_edge.png": "Photograph of a young man with light brown hair jumping mid-air off a large, reddish-brown rock. He's wearing a navy blue sweater, light blue shirt, gray pants, and brown shoes. His arms are outstretched, and he has a slight smile on his face. The background features a cloudy sky and a distant, leafless tree line. The grass around the rock is patchy.",
        "canny.png": "Aesthetics art, traditional asian pagoda, elaborate golden accents, sky blue and white color palette, swirling cloud pattern, digital illustration, east asian architecture, ornamental rooftop, intricate detailing on building, cultural representation.",
        "depth.png": "A swanky, minimalist living room with a huge floor-to-ceiling window letting in loads of natural light. A beige couch with white cushions sits on a wooden floor, with a matching coffee table in front. The walls are a soft, warm beige, decorated with two framed botanical prints. A potted plant chills in the corner near the window. Sunlight pours through the leaves outside, casting cool shadows on the floor.",
        "pose.png": "Photograph of a young man with light brown hair and a beard, wearing a beige flat cap, black leather jacket, gray shirt, brown pants, and white sneakers. He's sitting on a concrete ledge in front of a large circular window, with a cityscape reflected in the glass. The wall is cream-colored, and the sky is clear blue. His shadow is cast on the wall.",
    }
    controlnet_conditioning_scale = 1.0

    output_dir = f'examples_Pruning+ControlNet'
    os.makedirs(output_dir, exist_ok=True)
    
    for path in glob.glob('conds/*'):
        control_image = load_image(path)
        image_name = path.split('/')[-1]
        if image_name in prompt_dict:
            image = pipe(
                prompt=prompt_dict[image_name],
                negative_prompt=" ",
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=control_image.size[0],
                height=control_image.size[1],
                num_inference_steps=8,
                true_cfg_scale=4.0,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).images[0]
            image.save(os.path.join(output_dir, image_name))
