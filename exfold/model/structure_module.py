import sys

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from exfold.model.primitives import Linear, LayerNorm
from exfold.model.ipa import InvariantPointAttention

from exfold.utils.geometry.rigid_ops import Rigid3Array
from exfold.utils.geometry.vector_ops import Vec3Array
from exfold.utils.geometry.rotation_ops import Rot3Array
from exfold.utils.tensor_utils import dict_multimap
from exfold.utils.feats import backbone_and_literature_positions_to_atom3_pos

from exfold.common import nucleic_constants as nc


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden: int):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(
        self, 
        c_in: int, 
        c_hidden: int, 
        no_blocks: int, 
        no_angles: int, 
        epsilon: float
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """
    def __init__(self, c_hidden: int, use_high_precision: bool):
        super().__init__()
        self.c_hidden = c_hidden
        self.use_high_precision = use_high_precision

        # Multimer requires this to be run with fp32 precision during training
        precision = torch.float32 if self.use_high_precision else None

        self.linear = Linear(self.c_hidden, 6, init="final", precision=precision)
    
    def forward(self, activations: torch.Tensor) -> Rigid3Array:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res] Rigid
        """
        rigid_flat = self.linear(activations)
        
        rigid_flat = torch.unbind(rigid_flat, dim=-1)
        qx, qy, qz = rigid_flat[:3]
        qw = torch.ones_like(qx)
        translation = rigid_flat[3:]

        rotation = Rot3Array.from_quaternion(
            qw, qx, qy, qz, normalize=True,
        )
        translation = Vec3Array(*translation)
        return Rigid3Array(rotation, translation)


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c: int):
        super().__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(
        self, 
        c: int, 
        num_layers: int, 
        dropout_rate: float
    ):
        super().__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        trans_scale_factor,
        epsilon,
        inf,
        use_high_precision=False,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf
        self.use_high_precision = use_high_precision

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            use_high_precision=self.use_high_precision,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s, self.use_high_precision)

        # self.angle_resnet = AngleResnet(
        #     self.c_s,
        #     self.c_resnet,
        #     self.no_resnet_blocks,
        #     self.no_angles,
        #     self.epsilon,
        # )

    def forward(
        self,
        evoformer_output_dict: Dict[str, torch.Tensor],
        restype: torch.Tensor,
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
        _offload_inference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
                frames: 
                    [no_blocks, *, N_res, 4, 4]
                positions: C, P, N
                    [no_blocks, *, N_res, rna_backbone_atom_num, 3]
                # states:
                #     [no_blocks, *, N_res, C_s]
                # single:
                #     [*, N_res, C_s]
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N_res]
            mask = s.new_ones(s.shape[:-1])

        # [*, N_res, C_s]
        s = self.layer_norm_s(s)  # line 1

        # [*, N_res, N_res, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])  # line 2

        # 如果cpu_offloading, 则使用z_reference_list, 否则使用z
        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N_res, C_s]
        # s_initial = s  # 如果没有预测扭转角，则不需要s_initial
        s = self.linear_in(s)  # line 3

        # [*, N_res]
        rigids = Rigid3Array.identity(
            s.shape[:-1], 
            torch.float32 if self.use_high_precision else s.dtype,
            s.device, 
        )  # line4
        outputs = []
        for _ in range(self.no_blocks):
            # [*, N_res, C_s]
            s = s + self.ipa(
                s, 
                z, 
                rigids, 
                mask, 
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference, 
                _z_reference_list=z_reference_list
            )  # line 6
            s = self.ipa_dropout(s)  # line 7
            s = self.layer_norm_ipa(s)  # line 7
            s = self.transition(s)  # lines 8-9

            # [*, N_res]
            rigids: Rigid3Array = rigids @ self.bb_update(s)  # line 10
            
            # 下面这块是alphafold原本的逻辑
            # 1. 预测扭转角，得到的扭转角用于frame的生成。
            # # [*, N, 7, 2]
            # unnormalized_angles, angles = self.angle_resnet(s, s_initial)
            # 2. 根据扭转角和backbone frame计算出所有侧链frame。
            # all_frames_to_global = self.torsion_angles_to_frames(
            #     rigids.scale_translation(self.trans_scale_factor),
            #     angles,
            #     aatype,
            # )
            # 3. 将所有frame（backbone和侧链）作用于先验知识得到所有原子坐标。还是那句话，本质上是预测frame，先验知识就是这些原子在frame上的坐标。
            # pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            #     all_frames_to_global,
            #     aatype,
            # )

            pred_xyz = self.backbone_and_literature_positions_to_atom3_pos(
                rigids.scale_translation(self.trans_scale_factor),
                restype
            )

            preds = {
                "frames": rigids.scale_translation(self.trans_scale_factor).to_tensor4x4(),
                "positions": pred_xyz,
            }

            preds = {k: v.to(dtype=s.dtype) for k, v in preds.items()}

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()
        
        del z, z_reference_list

        if _offload_inference:
            evoformer_output_dict["pair"] = (
                evoformer_output_dict["pair"].to(s.device)
            )

        outputs = dict_multimap(torch.stack, outputs)
        # outputs["single"] = s

        return outputs

    def _init_residue_constants(
        self, 
        float_dtype: torch.dtype, 
        device: torch.device
    ):
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                nc.rna_restype_atom3_backbone_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
    
    def backbone_and_literature_positions_to_atom3_pos(
        self,
        backbone_to_global: Rigid3Array,
        restype: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            backbone_to_global: [*, N_res] Rigid3Array
            restype: [*, N_res]
        Returns:
            [*, N_res, rna_backbone_atom_num, 3]
        """
        self._init_residue_constants(backbone_to_global.dtype, backbone_to_global.device)
        return backbone_and_literature_positions_to_atom3_pos(
            backbone_to_global,
            restype,
            self.lit_positions,
        )
