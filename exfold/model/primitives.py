from importlib.util import find_spec
import math
from typing import Optional, Callable, List, Tuple
import numpy as np

deepspeed_is_installed = find_spec("deepspeed") is not None
ds4s_is_installed = deepspeed_is_installed and find_spec("deepspeed.ops.deepspeed4science") is not None
if deepspeed_is_installed:
    import deepspeed

if ds4s_is_installed:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

fa_is_installed = find_spec("flash_attn") is not None
if fa_is_installed:
    from flash_attn.bert_padding import unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func

import torch
import torch.nn as nn
from scipy.stats import truncnorm

from exfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


DEFAULT_LMA_Q_CHUNK_SIZE = 1024
DEFAULT_LMA_KV_CHUNK_SIZE = 4096
SOFTPLUS_INVERSE_1 = 0.541324854612918


# region: init_fn utils
def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        weights.fill_(SOFTPLUS_INVERSE_1)
# endregion


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
        precision: Optional[torch.dtype] = None
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

        self.precision = precision

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.dtype
        deepspeed_is_initialized = (
                deepspeed_is_installed and
                deepspeed.comm.comm.is_initialized()
        )
        if self.precision is not None:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                return nn.functional.linear(input.to(dtype=self.precision),
                                            self.weight.to(dtype=self.precision),
                                            bias).to(dtype=d)

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return nn.functional.linear(input, self.weight.to(dtype=d), bias)

        return nn.functional.linear(input, self.weight, self.bias)


class LayerNorm(nn.Module):
    def __init__(
        self, 
        c_in: int, 
        eps: float = 1e-5
    ):
        super().__init__()
        
        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        d = x.dtype
        deepspeed_is_initialized = (
            deepspeed_is_installed and 
            deepspeed.comm.comm.is_initialized()
        )
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x, 
                    self.c_in, 
                    self.weight.to(dtype=d), 
                    self.bias.to(dtype=d), 
                    self.eps
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out


def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = (
        deepspeed_is_installed and 
        deepspeed.comm.comm.is_initialized()
    )
    if d is torch.bfloat16 and not deepspeed_is_initialized:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


def _attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    biases: List[torch.Tensor]
) -> torch.Tensor:
    """
    Args:
        query (torch.Tensor): [*, H, Q, C_hidden].
        key (torch.Tensor): [*, H, K, C_hidden].
        value (torch.Tensor): [*, H, K, C_hidden].
        biases (List[torch.Tensor]): A list of bias tensors to add to the attention scores.

    Returns:
        torch.Tensor: The output tensor after applying the attention scores, 
        with shape [*, H, Q, C_hidden].
    """
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


#* openfold 在这里写了一个函数_attention_chunked_trainable，以分块的方式计算注意力，
#* 也支持checkpointing。但是被fused attention kernel代替，理由如下：
# MSA attention with training-time chunking of the softmax computation.
# Saves memory in the extra MSA stack. Probably obviated by our fused 
# attention kernel, which is now used by default.


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self,
        q_x: torch.Tensor, 
        kv_x: torch.Tensor,
        apply_scale: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare query, key, and value tensors for self-attention computation.

        Args:
            q_x (torch.Tensor): [*, Q, C_q]
            kv_x (torch.Tensor): [*, K, C_k]
            apply_scale (bool, optional): Whether to apply scaling factor to queries. 
                Defaults to True.

        Returns:
            [*, H, Q/K, C_hidden]
        """
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(
        self,
        o: torch.Tensor, 
        q_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform final processing after self-attention.

        This method performs final processing after self-attention computation,
        including gating mechanism if applicable and linear transformation.

        Args:
            o: [*, Q, H, C_hidden] Output tensor from self-attention.
            q_x: [*, Q, C_q] Input tensor for queries.

        Returns:
            o: 
                [*, Q, C_q]
        """
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
        
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        lma_kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE,
        use_flash: bool = False,
        flash_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory-efficient attention kernel.
                If none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch 
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        # region: check args
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError(
                "If use_lma is specified, lma_q_chunk_size and "
                "lma_kv_chunk_size must be provided"
            )

        if use_flash and biases is not None:
            raise ValueError(
                "use_flash is incompatible with the bias option. For masking, "
                "use flash_mask instead"
            )

        attn_options = [use_deepspeed_evo_attention, use_lma, use_flash]
        if sum(attn_options) > 1:
            raise ValueError(
                "Choose at most one alternative attention algorithm"
            )
        # endregion
        if biases is None:
            biases = []
        
        # DeepSpeed attention kernel applies scaling internally
        # DeepSpeed 注意力计算已经包含了scaling，其他的不包含
        q, k, v = self._prep_qkv(q_x, kv_x,
                                 apply_scale=not use_deepspeed_evo_attention)

        if use_deepspeed_evo_attention:
            if len(biases) > 2:
                raise ValueError(
                    "If use_deepspeed_evo_attention is True, you may only "
                    "provide up to two bias terms"
                )
            o = _deepspeed_evo_attn(q, k, v, biases)
        elif use_lma:
            biases = [
                b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],)) 
                for b in biases
            ]
            o = _lma(q, k, v, biases, lma_q_chunk_size, lma_kv_chunk_size)
            o = o.transpose(-2, -3)
        elif use_flash:
            o = _flash_attn(q, k, v, flash_mask)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class GlobalAttention(nn.Module):
    """
    Prepare for Algorithm 19 MSA global column-wise gated self-attention
    """
    def __init__(
        self, 
        c_in: int, 
        c_hidden: int, 
        no_heads: int, 
        inf: float, 
        eps: float
    ):
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(
            c_in, c_hidden * no_heads, bias=False, init="glorot"
        )

        self.linear_k = Linear(
            c_in, c_hidden, bias=False, init="glorot",
        )
        self.linear_v = Linear(
            c_in, c_hidden, bias=False, init="glorot",
        )
        self.linear_g = Linear(c_in, c_hidden * no_heads, init="gating")
        self.linear_o = Linear(c_hidden * no_heads, c_in, init="final")

        self.sigmoid = nn.Sigmoid()

    def forward(
        self, 
        m: torch.Tensor, 
        mask: torch.Tensor,
        use_lma: bool = False,
    ) -> torch.Tensor:
        """
        只支持lma或是原生pytorch
        Args:
            m: 
                [*, N_res, N_seq, C_in]
            mask: 
                [*, N_res, N_seq]
        Returns:
            m:
                [*, N_res, N_seq, C_in]
        """
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + self.eps
        )

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= (self.c_hidden ** (-0.5))

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        bias = (self.inf * (mask - 1))[..., :, None, :]
        if not use_lma:
            # [*, N_res, H, N_seq]
            a = torch.matmul(
                q,
                k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
            )
            a += bias
            a = softmax_no_cast(a)

            # [*, N_res, H, C_hidden]
            o = torch.matmul(
                a,
                v,
            )
        else:
            o = _lma(
                q, 
                k, 
                v, 
                [bias], 
                DEFAULT_LMA_Q_CHUNK_SIZE, 
                DEFAULT_LMA_KV_CHUNK_SIZE
            )

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m


def _deepspeed_evo_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    """""
    Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

    Args:
        q:
            [*, H, Q, C_hidden] query data
        k:
            [*, H, K, C_hidden] key data
        v:
            [*, H, V, C_hidden] value data
        biases:
            List of biases that broadcast to [*, H, Q, K]
    Returns:
        o:
            [*, Q, H, C_hidden]
    """

    if not ds4s_is_installed:
        raise ValueError(
            "_deepspeed_evo_attn requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )

    def reshape_dims(x):
        no_batch_dims = len(x.shape[:-3])
        if no_batch_dims < 2:
            return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
        if no_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    # [*, Q/K, H, C_hidden]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # Reshape tensors to match expected input shape [B, N, Q/K, H, C_hidden]
    # for DS4Sci_EvoformerAttention() by adding or flattening batch dims as needed.
    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dims(q)
        k = reshape_dims(k)
        v = reshape_dims(v)
        biases = [reshape_dims(b) for b in biases]

    # DeepSpeed attn. kernel requires inputs to be type bf16 or fp16
    # Cast to bf16 so kernel can be used during inference
    orig_dtype = q.dtype
    if orig_dtype not in [torch.bfloat16, torch.float16]:
        o = DS4Sci_EvoformerAttention(q.to(dtype=torch.bfloat16),
                                      k.to(dtype=torch.bfloat16),
                                      v.to(dtype=torch.bfloat16),
                                      [b.to(dtype=torch.bfloat16) for b in biases])

        o = o.to(dtype=orig_dtype)
    else:
        o = DS4Sci_EvoformerAttention(q, k, v, biases)

    o = o.reshape(orig_shape)

    return o


def _lma(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    biases: List[torch.Tensor], 
    q_chunk_size: int, 
    kv_chunk_size: int,
) -> torch.Tensor:
    """
    low-memory attention (Rabe & Staats 2021)

    This function implements Low Memory Attention (LMA) with chunked computation,
    which allows for efficient attention calculation with reduced memory consumption.

    Args:
        q:
            [*, H, Q, C_hidden] query data
        k: 
            [*, H, K, C_hidden] key data
        v: 
            [*, H, V, C_hidden] value data
        biases (List[torch.Tensor]): List of bias tensors.
        q_chunk_size (int): Size of each query chunk.
        kv_chunk_size (int): Size of each key/value chunk.

    Returns:
        o:
            [*, H, Q, C_hidden]
    """
    no_q, no_kv = q.shape[-2], k.shape[-2]

    # [*, H, Q, C_hidden]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s: q_s + q_chunk_size, :]
        large_bias_chunks = [
            b[..., q_s: q_s + q_chunk_size, :] for b in biases
        ]

        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s: kv_s + kv_chunk_size, :]
            v_chunk = v[..., kv_s: kv_s + kv_chunk_size, :]
            small_bias_chunks = [
                b[..., kv_s: kv_s + kv_chunk_size] for b in large_bias_chunks
            ]

            a = torch.einsum(
                "...hqd,...hkd->...hqk", q_chunk, k_chunk,
            )
       
            for b in small_bias_chunks:
                a += b
        
            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum("...hvf,...hqv->...hqf", v_chunk, exp_a)
 
            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)

        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)

        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values = chunk_values * max_diffs.unsqueeze(-1)
        chunk_weights = chunk_weights * max_diffs

        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)

        q_chunk_out = all_values / all_weights

        o[..., q_s: q_s + q_chunk_size, :] = q_chunk_out

    return o


def _flash_attn(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    kv_mask: torch.Tensor
) -> torch.Tensor:
    """
    flash attention

    Args:
        q:
            [*, H, Q, C_hidden] query data
        k: 
            [*, H, K, C_hidden] key data
        v: 
            [*, H, V, C_hidden] value data
        kv_mask:
            attention mask

    Returns:
        o:
            [*, Q, H, C_hidden]
    """
    if not fa_is_installed:
        raise ValueError(
            "_flash_attn requires that FlashAttention be installed"
        )
   
    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype

    q = q.half()
    k = k.half()
    v = v.half()
    kv_mask = kv_mask.half()

    # [*, B, N, H, C]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]
    
    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])
    
    q_max_s = n
    q_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device
    )

    # [B_flat, N, 2, H, C]
    kv = torch.stack([k, v], dim=-3) 
    kv_shape = kv.shape
    
    # [B_flat, N, 2 * H * C]
    kv = kv.reshape(*kv.shape[:-3], -1) 
    
    kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(kv, kv_mask)
    kv_unpad = kv_unpad.reshape(-1, *kv_shape[-3:])
   
    out = flash_attn_unpadded_kvpacked_func(
        q,
        kv_unpad,
        q_cu_seqlens,
        kv_cu_seqlens,
        q_max_s,
        kv_max_s,
        dropout_p=0.,
        softmax_scale=1.,  # q has been scaled already
    )
  
    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c) 

    out = out.to(dtype=dtype)

    return out
