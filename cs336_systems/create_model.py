import torch
from dataclasses import dataclass, field, asdict
from cs336_basics.model import *
import yaml
import argparse
import cs336_basics.model






@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention_scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        with nvtx.range("applying attention mask"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    
    with nvtx.range("computing output"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")    
    return output
    
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


@dataclass
class ModelConfig:
    model_type: str = "standard"
    num_layers: int = 2
    num_heads: int = 2
    context_length: int = 512
    d_model: int = 64
    vocab_size: int = 32000
    d_ff : int = 170


def load_config(yaml_path: str):
    dict = yaml.safe_load(open(yaml_path))
    return ModelConfig(**dict.get("model", {}))


def build_model(
    config: ModelConfig,
    checkpoint_block_size: int | None = None
):
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=config.context_length,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads
    )
    model.checkpoint_block_size = checkpoint_block_size
    return model



def create_model(
    config,
    checkpoint_block_size: int | None = None
):
    return build_model(config, checkpoint_block_size=checkpoint_block_size)


