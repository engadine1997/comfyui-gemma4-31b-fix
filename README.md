# comfyui-gemma4-31b-fix

## What is this?

This is self-contained custom node containing a bug fix for ComfyUI which prevents loading Gemma4 31B type models.

## Well, what is the bug?

`Gemma4_31B_Config` in `gemma4.py:84` never sets `num_global_key_value_heads` or `attention_k_eq_v`, which the 12B config does set. So the 10 global-attention layers were built with `16 KV heads × 512 = 8192` rows and a `v_proj`. Google's config has `4 heads × 512 = 2048` rows and no `v_proj`. The forward-pass code from PR `#14304` handles this correctly but the 31B config just never opted in.


## Installation

Clone the repo into your comfyui `custom_nodes` folder. *fin*. The `__init__.py` script loads the requesite fix.
