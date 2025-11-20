# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm

import copy

import gc

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def init_linear_least_squares(linear_layer: nn.Linear, A: torch.Tensor, B: torch.Tensor):
    """
    使用最小二乘法初始化 linear_layer 的 weight 和 bias，
    使得 linear_layer(A) ≈ B。

    Args:
        linear_layer: 要初始化的 nn.Linear 层
        A: 输入张量，shape = (N, in_features)
        B: 目标输出张量，shape = (N, out_features)
    """
    A = A.view(-1, A.shape[-1])
    B = B.view(-1, B.shape[-1])

    assert A.shape[0] == B.shape[0], "Batch size of A and B must match"
    assert A.shape[1] == linear_layer.in_features
    assert B.shape[1] == linear_layer.out_features

    # 增广输入：在最后一维加一列 1 用于偏置
    ones = torch.ones(A.shape[0], 1, device=A.device, dtype=A.dtype)
    A_aug = torch.cat([A, ones], dim=1)  # shape: (N, in_features + 1)

    # 使用 lstsq 求解最小二乘：A_aug @ X = B  => X = (in_features+1, out_features)
    # 注意：lstsq 默认求解 A @ X = B，所以这里 A_aug 是 (N, d+1), B 是 (N, out)
    # 返回的 solution 是 (d+1, out_features)
    solution = torch.linalg.lstsq(A_aug.float(), B.float()).solution  # shape: (in_features + 1, out_features)

    # 分离 weight 和 bias
    weight = solution[:-1, :].T.to(A.dtype)  # shape: (out_features, in_features)
    bias = solution[-1, :].to(A.dtype)       # shape: (out_features,)

    # 写入 linear 层（注意：必须使用 .data 避免破坏计算图）
    linear_layer.weight.data.copy_(weight)
    linear_layer.bias.data.copy_(bias)

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def tsn_l1(student, teacher):
    with torch.no_grad():
        teacher = teacher.float()
        teacher_norm = torch.norm(teacher, p=2, dim=-1, keepdim=True)
        teacher = teacher / teacher_norm
    
    student = student.float()
    student_norm = torch.norm(student, p=2, dim=-1, keepdim=True)
    student = student / student_norm
    
    return F.l1_loss(student, teacher, reduction='none').sum(dim=-1).mean()

def decompose_v0(x, eps=1e-8):
    # 方向：标准化（类似 LayerNorm）
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    direction = (x - mean) / (std + eps)
    
    # 幅度：可以用 L2 norm 或 std 表示“整体强度”
    # magnitude = torch.norm(x, p=2, dim=-1, keepdim=True)  # (B, N, 1)
    magnitude = std  # 如果你只关心波动强度

    return direction, magnitude



def decompose(x, eps=1e-8):
    magnitude = torch.norm(x, p=2, dim=-1, keepdim=True)  # (B, ..., 1)
    direction = x / (magnitude + eps)
    return direction, magnitude


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(1024)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        # 是否使用 scale rope
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        frame, height, width = video_fhw
        rope_key = f"{frame}_{height}_{width}"

        if rope_key not in self.rope_cache:
            seq_lens = frame * height * width
            freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
            freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
            freqs_frame = freqs_pos[0][:frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
            if self.scale_rope:
                freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
                freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

            else:
                freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

            freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
            self.rope_cache[rope_key] = freqs.clone().contiguous()
        vid_freqs = self.rope_cache[rope_key]

        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]

        return vid_freqs, txt_freqs


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(self, x, mod_params):
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn: bool = False,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
  

        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        
        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        attn_hs = hidden_states
        attn_ehs = encoder_hidden_states

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output
        
        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states, txt_modulated, img_modulated, attn_ehs, attn_hs



class QwenImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def init_distill(
        self,
        # layer_indexs = [[5,7],[8,9],[11,12],[16,18],[19,20],[22,24],[25,27],[42,43],[45,47],[48,49],[52,53],[54,55],[56,57]],
        layer_indexs = [[3,4],[5,7],[8,10],[11,12],[15,24],[25,27],[29,30],[42,43],[45,47],[48,49],[52,53],[54,55],[56,57]],
        # ehs_same = [[13,15],[31,32],[33,34],[36,38],[40,41],[50,59]],
        ehs_same = [[13,14],[31,32],[33,34],[36,37],[40,41],[50,59]]
    ):

        block_list = []
        block_proj_list = []
        for layer_index in layer_indexs:
            start, end = layer_index
            block_list.append(copy.deepcopy(self.transformer_blocks[start]))
            block_proj_list.append(nn.Linear(3072, 3072, bias=True, dtype=torch.bfloat16))
            block_proj_list.append(nn.Linear(3072, 3072, bias=True, dtype=torch.bfloat16))

        ehs_proj_list = []
        attn_proj_list = []
        for ehs_same_item in ehs_same:
            num_ehs_same = ehs_same_item[1] - ehs_same_item[0]
            for layer_index in layer_indexs:
                if layer_index[0] >= ehs_same_item[0] and layer_index[1] <= ehs_same_item[1]:
                    num_ehs_same -= (layer_index[1] - layer_index[0])
            for i in range(num_ehs_same):
                attn_proj_list.append(nn.Linear(3072, 3072, bias=True, dtype=torch.bfloat16))
            ehs_proj_list.append(nn.Linear(3072, 3072, bias=True, dtype=torch.bfloat16))
        
        j = 0
        ffn_proj_list = []
        for i in range(len(self.transformer_blocks)):
            if j >= len(layer_indexs) or i < layer_indexs[j][0] or i >= layer_indexs[j][1]:
                ffn_proj_list.append(nn.Linear(3072, 3072, bias=True, dtype=torch.bfloat16))
                ffn_proj_list.append(nn.Linear(3072, 3072, bias=True, dtype=torch.bfloat16))
            if j < len(layer_indexs) and i == layer_indexs[j][1]:
                j += 1

        self.layer_indexs = layer_indexs
        self.transformer_distill_blocks = nn.ModuleList(block_list)

        self.transformer_distill_blocks.eval()
        self.transformer_distill_blocks.requires_grad_(False)

        self.ehs_same = ehs_same
        self.attn_proj_list = nn.ModuleList(attn_proj_list)
        self.attn_proj_list.train()
        self.attn_proj_list.requires_grad_(True)

        self.ehs_proj_list = nn.ModuleList(ehs_proj_list)
        self.ehs_proj_list.train()
        self.ehs_proj_list.requires_grad_(True)

        self.block_proj_list = nn.ModuleList(block_proj_list)
        self.block_proj_list.train()
        self.block_proj_list.requires_grad_(True)

        self.ffn_proj_list = nn.ModuleList(ffn_proj_list)
        self.ffn_proj_list.train()
        self.ffn_proj_list.requires_grad_(True)

        self.img_in.eval()
        self.txt_in.eval()
        self.norm_out.eval()
        self.proj_out.eval()
        self.txt_norm.eval()
        self.pos_embed.eval()
        self.time_text_embed.eval()
        self.transformer_blocks.eval()

        self.img_in.requires_grad_(False)
        self.txt_in.requires_grad_(False)
        self.norm_out.requires_grad_(False)
        self.proj_out.requires_grad_(False)
        self.txt_norm.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        self.time_text_embed.requires_grad_(False)
        self.transformer_blocks.requires_grad_(False)

    def clear_blocks(self):
        # block_list = []
        max_distill_block_index = self.layer_indexs[-1][1]
        # for index_block, block in enumerate(self.transformer_blocks):
        #     if index_block <= max_distill_block_index:
        #         block_list.append(block)
        self.transformer_blocks = self.transformer_blocks[0:(max_distill_block_index+1)]
        self.transformer_blocks.eval()
        self.transformer_blocks.requires_grad_(False)
        gc.collect()
        torch.cuda.empty_cache()

    # def init_distill_cnn(self):
    #     ehs_cnns = []
    #     hs_cnns = []
    #     for layer_index in self.layer_indexs:
    #         ehs_cnns.append(UniversalInvertedBottleneck1D(in_channels=3072, out_channels=3072))
    #         hs_cnns.append(UniversalInvertedBottleneck2D(in_channels=3072, out_channels=3072))
    #     self.ehs_cnns = nn.ModuleList(ehs_cnns)
    #     self.hs_cnns = nn.ModuleList(hs_cnns)

    def distill_forward_attn(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        with torch.no_grad():
            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0
    
            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)
            else:
                if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                    logger.warning(
                        "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                    )
    
            hidden_states = self.img_in(hidden_states)
    
            timestep = timestep.to(hidden_states.dtype)
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)
    
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
    
            temb = (
                self.time_text_embed(timestep, hidden_states)
                if guidance is None
                else self.time_text_embed(timestep, guidance, hidden_states)
            )
    
            image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        loss = 0
        loss_list = []
        distill_index = 0
        encoder_hidden_states_, hidden_states_ = None, None
        for index_block, block in enumerate(self.transformer_blocks):
            with torch.no_grad():
                input_encoder_hidden_states, input_hidden_states = encoder_hidden_states, hidden_states
                encoder_hidden_states, hidden_states, img_attn_output, txt_attn_output = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
            
            if distill_index >= len(self.layer_indexs):
                continue
            # print(f"{distill_index = }, {self.layer_indexs = }, {self.input_layer_indexs}")
            
            if self.layer_indexs[distill_index][0] == index_block:
                # print(f"distill block forward: {distill_index = }, {index_block = }")
                block = self.transformer_distill_blocks[distill_index]

                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    if encoder_hidden_states_ is not None and hidden_states_ is not None:
                        encoder_hidden_states_, hidden_states_, img_attn_output_, txt_attn_output_ = self._gradient_checkpointing_func(
                            block,
                            hidden_states_,
                            input_encoder_hidden_states,
                            encoder_hidden_states_mask,
                            temb,
                            image_rotary_emb,
                        )
                    else:
                        encoder_hidden_states_, hidden_states_, img_attn_output_, txt_attn_output_ = self._gradient_checkpointing_func(
                            block,
                            input_hidden_states,
                            input_encoder_hidden_states,
                            encoder_hidden_states_mask,
                            temb,
                            image_rotary_emb,
                        )

                else:

                    if encoder_hidden_states_ is not None and hidden_states_ is not None:
                        encoder_hidden_states_, hidden_states_, img_attn_output_, txt_attn_output_ = block(
                            hidden_states=hidden_states_,
                            encoder_hidden_states=encoder_hidden_states_,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=attention_kwargs,
                        )
                    else:
                        encoder_hidden_states_, hidden_states_, img_attn_output_, txt_attn_output_ = block(
                            hidden_states=input_hidden_states,
                            encoder_hidden_states=input_encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=attention_kwargs,
                        )

            if self.layer_indexs[distill_index][1] == index_block:
                # print(f"compute loss: {distill_index = }, {index_block = }")
                # B, S, H = encoder_hidden_states_.shape
                
                with torch.no_grad():
                    img_attn_norm = torch.norm(img_attn_output, p=2, dim=-1, keepdim=True)
                    txt_attn_norm = torch.norm(txt_attn_output, p=2, dim=-1, keepdim=True)
                    img_attn_temp = img_attn_output / img_attn_norm
                    txt_attn_temp = txt_attn_output / txt_attn_norm
                # attention loss
                img_attn_norm_ = torch.norm(img_attn_output_, p=2, dim=-1, keepdim=True)
                txt_attn_norm_ = torch.norm(txt_attn_output_, p=2, dim=-1, keepdim=True)
                img_attn_temp_ = img_attn_output_ / img_attn_norm_
                txt_attn_temp_ = txt_attn_output_ / txt_attn_norm_
                img_attn_loss = F.mse_loss(img_attn_temp_, img_attn_temp, reduction='none').sum(dim=-1).mean()
                txt_attn_loss = F.mse_loss(txt_attn_temp_, txt_attn_temp, reduction='none').sum(dim=-1).mean()


                # img_attn_loss = F.mse_loss(img_attn_output_, img_attn_output)
                # txt_attn_loss = F.mse_loss(txt_attn_output_, txt_attn_output)


                # 分解教师和学生
                # img_dir_t, img_mag_t = decompose(img_attn_output_) 
                # img_dir_s, img_mag_s = decompose(img_attn_output)
                # img_attn_loss = F.mse_loss(img_dir_s, img_dir_t) + 1/10000 * F.mse_loss(img_mag_s, img_mag_t)
                # txt_dir_t, txt_mag_t = decompose(txt_attn_output_) 
                # txt_dir_s, txt_mag_s = decompose(txt_attn_output)
                # txt_attn_loss = F.mse_loss(txt_dir_s, txt_dir_t) + 1/10000 * F.mse_loss(txt_mag_s, txt_mag_t)


                # img_dir_t, img_mag_t = decompose(img_attn_output_) 
                # img_dir_s, img_mag_s = decompose(img_attn_output)
                # img_attn_loss = F.mse_loss(img_dir_s, img_dir_t) ##+ 1/10e6 * F.mse_loss(img_mag_s, img_mag_t)
                # txt_dir_t, txt_mag_t = decompose(txt_attn_output_) 
                # txt_dir_s, txt_mag_s = decompose(txt_attn_output)
                # txt_attn_loss = F.mse_loss(txt_dir_s, txt_dir_t) ##+ 1/10e6 * F.mse_loss(txt_mag_s, txt_mag_t)


                loss_list.append((img_attn_loss.item(), txt_attn_loss.item()))
                loss += (img_attn_loss + txt_attn_loss)

                ## KL
                # T=4.0
                # student_tensor = F.softmax(encoder_hidden_states_/T, dim=-1)
                # teacher_tensor = F.softmax(encoder_hidden_states/T, dim=-1)
                # ehs_loss = F.kl_div(teacher_tensor.log(), student_tensor, reduction='batchmean')

                distill_index += 1
        # loss = loss / len(self.layer_indexs)
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, loss, loss_list)

        return Transformer2DModelOutput(sample=output), loss, loss_list


    @torch.no_grad()
    def init_linear(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        with torch.no_grad():
            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0
    
            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)
            else:
                if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                    logger.warning(
                        "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                    )
    
            hidden_states = self.img_in(hidden_states)
    
            timestep = timestep.to(hidden_states.dtype)
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)
    
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
    
            temb = (
                self.time_text_embed(timestep, hidden_states)
                if guidance is None
                else self.time_text_embed(timestep, guidance, hidden_states)
            )
    
            image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        loss = 0
        loss_list = []
        distill_index = 0
        attn_proj_index = 0
        ffn_proj_index = 0
        ehs_same_index = 0

        ehs_proj_index = 0
        block_proj_index = 0

        input_encoder_hidden_states_0, input_hidden_states_0 = None, None
        encoder_hidden_states_0, hidden_states_0 = None, None
        txt_modulated_0, img_modulated_0 = None, None
        attn_ehs_0, attn_hs_0 = None, None
        txt_modulated_ = None
        encoder_hidden_states_ = None

        for index_block, block in enumerate(self.transformer_blocks):

            with torch.no_grad():
                input_encoder_hidden_states, input_hidden_states = encoder_hidden_states, hidden_states
                encoder_hidden_states, hidden_states, txt_modulated, img_modulated, attn_ehs, attn_hs = block(
                    hidden_states=input_hidden_states,
                    encoder_hidden_states=input_encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
                
            if distill_index < len(self.layer_indexs) and self.layer_indexs[distill_index][0] == index_block:
                block = self.transformer_distill_blocks[distill_index]
                # use_hs_0 = distill_index > 0 and self.layer_indexs[distill_index][0] == self.layer_indexs[distill_index -1][1] + 1

                with torch.no_grad():
                    encoder_hidden_states_0, hidden_states_0, txt_modulated_0, img_modulated_0, attn_ehs_0, attn_hs_0 = block(
                        hidden_states=input_hidden_states,
                        encoder_hidden_states=input_encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )
                    input_encoder_hidden_states_0, input_hidden_states_0 = input_encoder_hidden_states, input_hidden_states
            
            if distill_index < len(self.layer_indexs) and self.layer_indexs[distill_index][1] == index_block:
                init_linear_least_squares(self.block_proj_list[block_proj_index + 0], encoder_hidden_states_0, encoder_hidden_states - encoder_hidden_states_0)
                init_linear_least_squares(self.block_proj_list[block_proj_index + 1], hidden_states_0, hidden_states - hidden_states_0)
                block_proj_index += 2

            use_txt = ehs_same_index < len(self.ehs_same) and index_block > self.ehs_same[ehs_same_index][0] and index_block <= self.ehs_same[ehs_same_index][1]
            use_ehs_proj = use_txt and index_block == self.ehs_same[ehs_same_index][1]
            if ehs_same_index < len(self.ehs_same):
                if index_block == self.ehs_same[ehs_same_index][0]:
                    txt_modulated_ = txt_modulated
                    encoder_hidden_states_ = encoder_hidden_states
                elif index_block == self.ehs_same[ehs_same_index][1]:
                    ehs_same_index += 1
            
            if use_txt:
                if (distill_index >= len(self.layer_indexs)) or (self.layer_indexs[distill_index][0] >= index_block or self.layer_indexs[distill_index][1] < index_block):
                    init_linear_least_squares(self.attn_proj_list[attn_proj_index], txt_modulated_, txt_modulated - txt_modulated_)
                    # x = self.attn_proj_list[attn_proj_index](txt_modulated_) + txt_modulated_
                    # ehs_loss = tsn_l1(x, txt_modulated)
                    # loss_list.append((ehs_loss.item(), index_block, attn_proj_index, "txt_modulated"))
                    # loss += ehs_loss
                    attn_proj_index += 1
                if use_ehs_proj:
                    init_linear_least_squares(self.ehs_proj_list[ehs_proj_index], encoder_hidden_states_, encoder_hidden_states - encoder_hidden_states_)
                    ehs_proj_index += 1

            
            use_ffn = distill_index >= len(self.layer_indexs) or index_block < self.layer_indexs[distill_index][0] or index_block >= self.layer_indexs[distill_index][1]
            if use_ffn:
                ehs_ffn_proj = self.ffn_proj_list[ffn_proj_index + 0]
                hs_ffn_proj = self.ffn_proj_list[ffn_proj_index + 1]
                use_distill_block = distill_index < len(self.layer_indexs) and index_block >= self.layer_indexs[distill_index][0] and index_block <= self.layer_indexs[distill_index][1]
                ehs_input = attn_ehs_0 if use_distill_block else attn_ehs
                hs_input = attn_hs_0 if use_distill_block else attn_hs
               
                init_linear_least_squares(ehs_ffn_proj, ehs_input, encoder_hidden_states - ehs_input)
                init_linear_least_squares(hs_ffn_proj, hs_input, hidden_states - hs_input)

                # ehs_ffn = ehs_ffn_proj(ehs_input) + ehs_input
                # hs_ffn = hs_ffn_proj(hs_input) + hs_input
                # ehs_loss = tsn_l1(ehs_ffn, encoder_hidden_states)
                # hs_loss = tsn_l1(hs_ffn, hidden_states)
                # loss_list.append((ehs_loss.item(), index_block, ffn_proj_index + 0, "ehs_ffn_proj "))
                # loss_list.append((hs_loss.item(), index_block, ffn_proj_index + 1, "hs_ffn_proj  "))
                # loss += ehs_loss
                # loss += hs_loss
                ffn_proj_index += 2
            
            if distill_index < len(self.layer_indexs) and self.layer_indexs[distill_index][1] == index_block:
                distill_index += 1

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, loss, loss_list)

        return Transformer2DModelOutput(sample=output), loss, loss_list

    def distill_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        with torch.no_grad():
            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0
    
            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)
            else:
                if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                    logger.warning(
                        "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                    )
    
            hidden_states = self.img_in(hidden_states)
    
            timestep = timestep.to(hidden_states.dtype)
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)
    
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
    
            temb = (
                self.time_text_embed(timestep, hidden_states)
                if guidance is None
                else self.time_text_embed(timestep, guidance, hidden_states)
            )
    
            image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        loss = 0
        loss_list = []
        distill_index = 0
        attn_proj_index = 0
        ffn_proj_index = 0
        ehs_same_index = 0

        ehs_proj_index = 0
        block_proj_index = 0

        input_encoder_hidden_states_0, input_hidden_states_0 = None, None
        encoder_hidden_states_0, hidden_states_0 = None, None
        txt_modulated_0, img_modulated_0 = None, None
        attn_ehs_0, attn_hs_0 = None, None
        txt_modulated_ = None
        encoder_hidden_states_ = None

        for index_block, block in enumerate(self.transformer_blocks):

            with torch.no_grad():
                input_encoder_hidden_states, input_hidden_states = encoder_hidden_states, hidden_states
                encoder_hidden_states, hidden_states, txt_modulated, img_modulated, attn_ehs, attn_hs = block(
                    hidden_states=input_hidden_states,
                    encoder_hidden_states=input_encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
                
            if distill_index < len(self.layer_indexs) and self.layer_indexs[distill_index][0] == index_block:
                block = self.transformer_distill_blocks[distill_index]
                # use_hs_0 = distill_index > 0 and self.layer_indexs[distill_index][0] == self.layer_indexs[distill_index -1][1] + 1

                with torch.no_grad():
                    encoder_hidden_states_0, hidden_states_0, txt_modulated_0, img_modulated_0, attn_ehs_0, attn_hs_0 = block(
                        hidden_states=input_hidden_states,
                        encoder_hidden_states=input_encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )
                    input_encoder_hidden_states_0, input_hidden_states_0 = input_encoder_hidden_states, input_hidden_states

            if distill_index < len(self.layer_indexs) and self.layer_indexs[distill_index][1] == index_block:
                ehs = self.block_proj_list[block_proj_index + 0](encoder_hidden_states_0) + encoder_hidden_states_0
                hs = self.block_proj_list[block_proj_index + 1](hidden_states_0) + hidden_states_0
                ehs_loss = tsn_l1(ehs, encoder_hidden_states)
                hs_loss = tsn_l1(hs, hidden_states)
                loss_list.append((ehs_loss.item(), index_block, block_proj_index + 0, "ehs_block_proj"))
                loss_list.append(( hs_loss.item(), index_block, block_proj_index + 1, " hs_block_proj"))
                loss += ehs_loss + hs_loss
                block_proj_index += 2

            use_txt = ehs_same_index < len(self.ehs_same) and index_block > self.ehs_same[ehs_same_index][0] and index_block <= self.ehs_same[ehs_same_index][1]
            use_ehs_proj = use_txt and index_block == self.ehs_same[ehs_same_index][1]
            if ehs_same_index < len(self.ehs_same):
                if index_block == self.ehs_same[ehs_same_index][0]:
                    txt_modulated_ = txt_modulated
                    encoder_hidden_states_ = encoder_hidden_states
                elif index_block == self.ehs_same[ehs_same_index][1]:
                    ehs_same_index += 1
            
            if use_txt:
                if (distill_index >= len(self.layer_indexs)) or (self.layer_indexs[distill_index][0] >= index_block or self.layer_indexs[distill_index][1] < index_block):
                    x = self.attn_proj_list[attn_proj_index](txt_modulated_) + txt_modulated_
                    ehs_loss = tsn_l1(x, txt_modulated)
                    loss_list.append((ehs_loss.item(), index_block, attn_proj_index, " txt_modulated"))
                    loss += ehs_loss
                    attn_proj_index += 1

                if use_ehs_proj:
                    ehs = self.ehs_proj_list[ehs_proj_index](encoder_hidden_states_) + encoder_hidden_states_
                    ehs_loss = tsn_l1(ehs, encoder_hidden_states)
                    loss_list.append((ehs_loss.item(), index_block, ehs_proj_index, "      ehs_proj"))
                    ehs_proj_index += 1 
            
            use_ffn = distill_index >= len(self.layer_indexs) or index_block < self.layer_indexs[distill_index][0] or index_block >= self.layer_indexs[distill_index][1]
            if use_ffn:
                ehs_ffn_proj = self.ffn_proj_list[ffn_proj_index + 0]
                hs_ffn_proj = self.ffn_proj_list[ffn_proj_index + 1]
                use_distill_block = distill_index < len(self.layer_indexs) and index_block >= self.layer_indexs[distill_index][0] and index_block <= self.layer_indexs[distill_index][1]
                ehs_input = attn_ehs_0 if use_distill_block else attn_ehs
                hs_input = attn_hs_0 if use_distill_block else attn_hs
                ehs_ffn = ehs_ffn_proj(ehs_input) + ehs_input
                hs_ffn = hs_ffn_proj(hs_input) + hs_input
                ehs_loss = tsn_l1(ehs_ffn, encoder_hidden_states)
                hs_loss = tsn_l1(hs_ffn, hidden_states)
                loss_list.append((ehs_loss.item(), index_block, ffn_proj_index + 0, "  ehs_ffn_proj"))
                loss_list.append((hs_loss.item(), index_block, ffn_proj_index + 1, "   hs_ffn_proj"))
                loss += ehs_loss
                loss += hs_loss
                ffn_proj_index += 2
            
            if distill_index < len(self.layer_indexs) and self.layer_indexs[distill_index][1] == index_block:
                distill_index += 1

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, loss, loss_list)

        return Transformer2DModelOutput(sample=output), loss, loss_list


    @torch.no_grad()
    def distill_infer(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        distill_index = 0
        for index_block, block in enumerate(self.transformer_blocks):
            use_block = distill_index >= len(self.layer_indexs)
            # if index_block == 20:
            #     use_block=True
            #     distill_index += 1
            
            if not use_block:
                start, end = self.layer_indexs[distill_index]
                use_block = start >= index_block or end < index_block
                if start == index_block:
                    block = self.transformer_distill_blocks[distill_index]
                if end == index_block:
                    distill_index += 1
            if use_block:
                encoder_hidden_states, hidden_states, _, _ = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)