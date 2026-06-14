"""
Attention module initialization
"""

from .vanilla import vanilla_attention
from .sdpa import sdpa_attention
from .fa2 import flash_attention_2
from .sparse import sparse_attention
from .sparse_int8 import sparse_int8_attention

__all__ = [
    'vanilla_attention',
    'sdpa_attention',
    'flash_attention_2',
    'sparse_attention',
    'sparse_int8_attention',
]
