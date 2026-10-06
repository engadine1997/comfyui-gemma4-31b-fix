"""Fixes loading of Gemma 4 31B text encoders (quantized or not).

ComfyUI's Gemma4_31B_Config doesn't describe the 10 global-attention layers: they use
4 KV heads (head_dim 512) with attention_k_eq_v (no v_proj), same as Gemma4_12B_Config.
Without it the layers are built with a 16-head k_proj (8192 rows) plus a v_proj, which
fails with "scale must have shape (8192, 336), got (2048, 336)" on quantized weights.

Applied at startup, so it needs no changes to ComfyUI itself. Does nothing once
upstream sets these values.
"""
from comfy.text_encoders.gemma4 import Gemma4_31B_Config

if not Gemma4_31B_Config.attention_k_eq_v:
    Gemma4_31B_Config.attention_k_eq_v = True
    Gemma4_31B_Config.num_global_key_value_heads = 4
    print("[gemma4_31b_fix] Patched Gemma4_31B_Config for global-attention layers (4 KV heads, k_eq_v)")

NODE_CLASS_MAPPINGS = {}
