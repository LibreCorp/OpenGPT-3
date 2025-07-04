"""
Core Transformer components for GPT-style language models.
"""
from opengpt3.modeling.attention import Past, AttentionLayer
from opengpt3.modeling.masking import PadMasking, FutureMasking, causal_mask, local_mask
from opengpt3.modeling.split_merge import split_heads, merge_heads
from opengpt3.modeling.transformer import TransformerBlock, GPTModel as Transformer
