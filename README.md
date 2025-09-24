<div align="center">
  <h1>Qwen-Image-Pruning</h1>
<a href='https://huggingface.co/OPPOer/Qwen-Image-Pruning'><img src='https://img.shields.io/badge/🤗%20HuggingFace-Qwen--Image--Pruning-ffd21f.svg'></a>
</div>

## Environment

``` sh
pip install torch
pip install git+https://github.com/huggingface/diffusers
```

## Inference

### 1. Qwen-Image-Pruning Inference

Download the checkpoint from [Qwen-Image-Pruning](https://huggingface.co/OPPOer/Qwen-Image-Pruning)

``` sh
python inference.py
python inference_12B.py # For Qwen-Image-12B Model Inference
```

### 2. Qwen-Image-Pruning & Realism-LoRA Inference

Download the checkpoint from [Qwen-Image-Pruning](https://huggingface.co/OPPOer/Qwen-Image-Pruning) and [Qwen-Image-Realism-LoRA](https://huggingface.co/flymy-ai/qwen-image-realism-lora)

``` sh
python inference_realism_lora.py
```

### 3. Qwen-Image-Pruning & ControlNet Inference

Download the checkpoint from [Qwen-Image-Pruning](https://huggingface.co/OPPOer/Qwen-Image-Pruning) and [Qwen-Image-ControlNet-Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union)

``` sh
python inference_controlnet.py
```
