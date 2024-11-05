import sys
import math
from typing import Optional, Tuple, Sequence, Union

import torch
import torch.nn as nn

from exfold.model.primitives import Linear, ipa_point_weights_init_

from exfold.utils.geometry.rigid_ops import Rigid3Array
from exfold.utils.geometry.vector_ops import Vec3Array
from exfold.utils.geometry import square_euclidean_distance
from exfold.utils.precision_utils import is_fp16_enabled
from exfold.utils.tensor_utils import flatten_final_dims


class PointProjection(nn.Module):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        use_high_precision: bool,
        return_local_points: bool = False,
    ):
        super().__init__()
        self.return_local_points = return_local_points
        self.no_heads = no_heads
        self.num_points = num_points
        self.use_high_precision = use_high_precision

        # Multimer requires this to be run with fp32 precision during training
        precision = torch.float32 if self.use_high_precision else None
        self.linear = Linear(c_hidden, no_heads * 3 * num_points, precision=precision)

    def forward(
        self, 
        activations: torch.Tensor, 
        rigids: Rigid3Array,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        相当于将激活值通过线性层预测在局部坐标系下的坐标, 在变换到全局坐标系下
        Args:
            activations: [*, c_hidden]
            rigids: [*]
        
        Returns
        -------
            points_global: [*, heads, num_points, 3]
            points_local: [*, heads, num_points, 3]
        """
        # TODO: Needs to run in high precision during training
        # [*, heads * 3 * num_points]
        points_local = self.linear(activations)
        # [*, heads, num_points, 3]
        out_shape = points_local.shape[:-1] + (self.no_heads, self.num_points, 3)
        # [*, heads, num_points * 3]
        if self.use_high_precision:
            points_local = points_local.view(
                points_local.shape[:-1] + (self.no_heads, -1)
            )
        # [*, heads * num_points] * 3 or [*, heads, num_points] * 3
        points_local = torch.split(
            points_local, points_local.shape[-1] // 3, dim=-1
        )
        # [*, heads, num_points, 3]
        points_local = torch.stack(points_local, dim=-1).view(out_shape)

        points_global = rigids[..., None, None].apply(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        use_high_precision: bool = False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.use_high_precision = use_high_precision

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads

        self.linear_q = Linear(self.c_s, hc, bias=not use_high_precision)
        self.linear_q_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            self.use_high_precision
        )

        self.linear_k = Linear(self.c_s, hc, bias=not use_high_precision)
        self.linear_v = Linear(self.c_s, hc, bias=not use_high_precision)
        self.linear_k_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            self.use_high_precision
        )

        self.linear_v_points = PointProjection(
            self.c_s,
            self.no_v_points,
            self.no_heads,
            self.use_high_precision
        )
        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-2)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid3Array,
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference and inplace_safe:
            z = _z_reference_list
        else:
            z = [z]

        # point_weights: w_C
        point_variance = (max(self.no_qk_points, 1) * 9.0 / 2)
        point_weights = math.sqrt(1.0 / point_variance)
        # head_weights: gamma_h, [H]
        head_weights = self.softplus(self.head_weights)
        # scalar_weights: sqrt(1/c)
        scalar_variance = max(self.c_hidden, 1) * 1.
        scalar_weights = math.sqrt(1.0 / scalar_variance)
        # logit_weights: sqrt(1/3)
        logit_weights = math.sqrt(1. / 3)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        a = self.linear_b(z[0])

        if _offload_inference:
            assert (sys.getrefcount(z[0]) == 2)
            z[0] = z[0].cpu()

        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        k = self.linear_k(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))

        q = q * scalar_weights

        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = a + torch.einsum('...qhc,...khc->...qkh', q.float(), k.float())
        else:
            a = a + torch.einsum('...qhc,...khc->...qkh', q, k)
        
        a = a.to(dtype=s.dtype)

        # [*, N_res, H, P_qk]
        q_pts = Vec3Array.from_array(self.linear_q_points(s, r))
        # [*, N_res, H, P_qk]
        k_pts = Vec3Array.from_array(self.linear_k_points(s, r))
        # [*, N_res, N_res, H, P_qk]
        pt_att = square_euclidean_distance(q_pts.unsqueeze(-3), k_pts.unsqueeze(-4), epsilon=0.)

        if inplace_safe:
            pt_att *= head_weights[..., None]
        else:
            pt_att = pt_att * head_weights[..., None]

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * point_weights * (-0.5)
        pt_att = pt_att.to(dtype=s.dtype)

        if inplace_safe:
            a += pt_att
            del pt_att
        else:
            a = a + pt_att

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        if inplace_safe:
            # [*, N_res, N_res, H]
            a += square_mask.unsqueeze(-1)
        else:
            # [*, N_res, N_res, H]
            a = a + square_mask.unsqueeze(-1)

        a = a * logit_weights  # Normalize by number of logit terms (3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H * C_hidden]
        v = self.linear_v(s)
        # [*, N_res, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        # [*, N_res, H, C_hidden]
        o = torch.einsum('...qkh, ...khc->...qhc', a, v)
        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, H, P_v]
        v_pts = Vec3Array.from_array(self.linear_v_points(s, r))

        # [*, N_res, N_res, H, P_v]
        o_pt: Vec3Array = v_pts.unsqueeze(-4) * a.unsqueeze(-1)
        # [*, N_res, H, P_v]
        o_pt = o_pt.sum(dim=-3)
        # [*, N_res, H * P_v]
        o_pt = o_pt.reshape(o_pt.shape[:-2] + (-1,))

        # [*, N_res, H * P_v]
        # 此时o_pt的dtype可能和a不一致 
        o_pt = r[..., None].apply_inverse_to_point(o_pt)
        o_pt_flat = [o_pt.x, o_pt.y, o_pt.z]
        # [*, N_res, H * P_v] * 3
        o_pt_flat = [x.to(dtype=a.dtype) for x in o_pt_flat]

        # [*, N_res, H * P_v]
        o_pt_norm = o_pt.norm(epsilon=1e-8)

        if _offload_inference:
            z[0] = z[0].to(o_pt.x.device)

        # [*, N_res, H, C_z]
        o_pair = torch.einsum('...ijh, ...ijc->...ihc', a, z[0].to(dtype=a.dtype))
        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *o_pt_flat, o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s
