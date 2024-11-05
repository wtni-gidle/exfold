from typing import Tuple, Optional
import torch
import torch.nn as nn

from exfold.model.primitives import Linear, LayerNorm
from exfold.utils.tensor_utils import (
    add, 
    one_hot
)



class SeqEmbedder(nn.Module):
    """
    An embedder that processes only single sequence.
    Despite this, the input still includes msa_feat (with only one seq).
    """
    def __init__(
        self, 
        tf_dim: int,
        msa_dim: int,
        c_m: int,
        c_z: int,
        relpos_k: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_m = c_m
        self.c_z = c_z

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)
    
    def relpos(self, ri: torch.Tensor) -> torch.Tensor:
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N_res]
        Returns:
            [*, N_res, N_res, c_z]
        """
        # [*, N_res, N_res]
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        # [*, N_res, N_res, no_bins]
        d = one_hot(d, boundaries)
        d = d.to(ri.dtype)
        return self.linear_relpos(d)
    
    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf (target_feat):
                Features of shape [*, N_res, tf_dim]
            ri (residue_index):
                Features of shape [*, N_res]
            msa (msa_feat):
                Features of shape [*, 1, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, 1, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding
        """
        # [*, 1, N_res, c_m]
        msa_emb = self.linear_msa_m(msa) + self.linear_tf_m(tf).unsqueeze(-3)

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(
            pair_emb, 
            tf_emb_i[..., None, :], 
            inplace=inplace_safe
        )
        pair_emb = add(
            pair_emb, 
            tf_emb_j[..., None, :, :], 
            inplace=inplace_safe
        )

        return msa_emb, pair_emb


class SSEmbedder(nn.Module):
    def __init__(
        self, 
        ss_dim: int, 
        c_z: int,
        **kwargs,
    ):
        """
        Args:
            ss_dim:
                Final dimension of the ss features
            c_z:
                Pair embedding dimension
        """
        super().__init__()
        self.ss_dim = ss_dim
        self.c_z = c_z
        self.ss_linear = Linear(self.ss_dim, self.c_z)

    def forward(self, ss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ss: 
                [*, N_res, N_res, ss_dim]
        Returns:
            pair_emb_ss: 
                [*, N_res, N_res, c_z]
        """
        pair_emb_ss = self.ss_linear(ss)
        
        return pair_emb_ss


#todo 加入语言模型的embedding可以参考rhofold和openfold的preembedding

#todo need to try
class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        with_x: bool = True,
        inf: float = 1e8,
        **kwargs,
    ):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.with_x = with_x
        self.inf = inf
        
        #* For with_x=False,  e.g. is_geom=True, we do not embed x
        if self.with_x:
            self.linear = Linear(self.no_bins, self.c_z)
        
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_res, c_m] First row of the MSA embedding.
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted glycos_N coordinates
        Returns:
            m_update:
                [*, N_res, C_m] MSA embedding update
            z_update:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N_res, C_m]
        m_update = self.layer_norm_m(m)
        if inplace_safe:
            m.copy_(m_update)
            m_update = m
        
        # [*, N_res, N_res, C_z]
        z_update = self.layer_norm_z(z)
        if inplace_safe:
            z.copy_(z_update)
            z_update = z
        
        if self.with_x:
            # This squared method might become problematic in FP16 mode.
            # [no_bins]
            bins = torch.linspace(
                self.min_bin,
                self.max_bin,
                self.no_bins,
                device=x.device,
                requires_grad=False,
            )
            squared_bins = (bins ** 2)
            upper = torch.cat(
                [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
            )
            # [*, N_res, N_res, 1]
            d = torch.sum(
                (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
            )

            # [*, N_res, N_res, no_bins]
            d = ((d > squared_bins) * (d < upper)).type(x.dtype)

            # [*, N_res, N_res, C_z]
            d = self.linear(d)
            z_update = add(z_update, d, inplace_safe)

        return m_update, z_update


class RecyclingEmbedderWithoutX(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        **kwargs,
    ):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_res, c_m] First row of the MSA embedding.
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted glycos_N coordinates
        Returns:
            m_update:
                [*, N_res, C_m] MSA embedding update
            z_update:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N_res, C_m]
        m_update = self.layer_norm_m(m)
        if inplace_safe:
            m.copy_(m_update)
            m_update = m
        
        # [*, N_res, N_res, C_z]
        z_update = self.layer_norm_z(z)
        if inplace_safe:
            z.copy_(z_update)
            z_update = z

        return m_update, z_update
