from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from exfold.model.primitives import Linear
from exfold.utils.chunk_utils import chunk_layer
from exfold.utils.precision_utils import is_fp16_enabled
from exfold.utils.tensor_utils import flatten_final_dims


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """
    def __init__(
        self, 
        c_m: int, 
        c_z: int, 
        c_hidden: int, 
        eps: float = 1e-3
    ):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")

    def _opm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: [*, N_res, N_seq, C_hidden]
            b: [*, N_res, N_seq, C_hidden]
        Returns:
            outer: [*, N_res, N_res, C_z]
        """
        # [*, N_res, N_res, C_hidden, C_hidden]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C_hidden * C_hidden]
        outer = flatten_final_dims(outer, 2)

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    def _chunk(
        self, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        chunk_size: int
    ) -> torch.Tensor:
        """
        Args:
            a: [*, N_res, N_seq, C_hidden]
            b: [*, N_res, N_seq, C_hidden]
        Returns:
            outer: [*, N_res, N_res, C_z]
        """
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            # [N_res, N_res, C_z]
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def _forward(
        self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C_hidden]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) 
        a = a * mask
        
        b = self.linear_2(ln) 
        b = b * mask

        del ln
        # [*, N_res, N_seq, C_hidden]
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        if inplace_safe:
            outer /= norm
        else:
            outer = outer / norm

        return outer

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(m.float(), mask, chunk_size, inplace_safe)
        else:
            return self._forward(m, mask, chunk_size, inplace_safe)
