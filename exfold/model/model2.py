from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from exfold.model.embedders import (
    SeqEmbedder, 
    SSEmbedder, 
    RecyclingEmbedder,
    RecyclingEmbedderWithoutX
)
from exfold.model.evoformer import EvoformerStack
from exfold.model.structure_module import StructureModule
from exfold.model.heads import End2EndHeads, GeometryHeads

from exfold.common import nucleic_constants as nc
from exfold.utils.feats import backbone_atom_fn
from exfold.utils.tensor_utils import add, tensor_tree_map


class ExFoldGeometry(nn.Module):
    def __init__(self, config) -> None:
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super().__init__()

        self.globals = config.globals
        self.config = config.model

        self.input_embedder = SeqEmbedder(
            **self.config["seq_embedder"]
        )

        self.ss_embedder = SSEmbedder(
            **self.config["ss_embedder"]
        )

        self.recycling_embedder = RecyclingEmbedderWithoutX(
            **self.config["recycling_embedder"]
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
            output_s=False
        )
        
        self.heads = GeometryHeads(
            self.config["geom_heads"],
        )
    
    def iteration(
        self, 
        feats: Dict[str, torch.Tensor], 
        prevs: List[torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        prevs: [m_1_prev, z_prev]
        Returns:
            outputs: Dict[str, torch.Tensor]
                msa: 
                    [*, N_seq, N_res, C_m]
                pair: 
                    [*, N_res, N_res, C_z]
        """
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)
        
        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        n_res = feats["target_feat"].shape[-2]

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize the MSA and pair representations
        # m: [*, N_seq, N_res, C_m] N_seq=1 for no msa mode
        # z: [*, N_res, N_res, C_z]
        m, z = self.input_embedder(
            tf=feats["target_feat"],
            ri=feats["residue_index"],
            msa=feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

        # ss_emb: [*, N_res, N_res, c_z]
        ss_emb = self.ss_embedder(
            ss=feats["ss"],
        )
        # [*, N_res, N_res, C_z]
        z = add(z, ss_emb, inplace=inplace_safe)

        del ss_emb

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev = reversed([prevs.pop() for _ in range(3)])

        # Initialize the recycling embeddings, if needs be 
        if m_1_prev is None: 
            # [*, N_res, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n_res, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N_res, N_res, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n_res, n_res, self.config.input_embedder.c_z),
                requires_grad=False,
            )

        # The recycling embedder is memory-intensive, so we offload first
        if self.globals.offload_inference and inplace_safe:
            m = m.cpu()
            z = z.cpu()
        
        # m_1_prev_emb: [*, N_res, C_m]
        # z_prev_emb: [*, N_res, N_res, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m=m_1_prev,
            z=z_prev,
            inplace_safe=inplace_safe,
        )

        if self.globals.offload_inference and inplace_safe:
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, N_seq, N_res, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N_res, N_res, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, N_seq, N_res, C_m] N_seq=1 for no msa mode
        # z: [*, N_res, N_res, C_z]
        # s: [*, N_res, C_s]
        if self.globals.offload_inference:
            input_tensors = [m, z]
            del m, z
            m, z, _ = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )

            del input_tensors
        else:
            m, z, _ = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m
        outputs["pair"] = z

        del z

        # Save embeddings for use during the next recycling iteration

        # [*, N_res, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N_res, N_res, C_z]
        z_prev = outputs["pair"]

        return outputs, m_1_prev, z_prev

    def _disable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = (
            self.config.evoformer_stack.blocks_per_ckpt
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.
            outputs:
                msa: 
                    [*, N_seq, N_res, C_m]
                pair: 
                    [*, N_res, N_res, C_z]
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev = None, None
        prevs = [m_1_prev, z_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["restype"].shape[-1]
        num_recycles = 0
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev = self.iteration(
                    feats,
                    prevs,
                )

                if not is_final_iter:
                    num_recycles += 1

                    del outputs
                    prevs = [m_1_prev, z_prev]
                    del m_1_prev, z_prev
                else:
                    break

        # Run auxiliary heads
        outputs.update(self.heads(outputs))

        return outputs


class ExFoldEnd2End(nn.Module):
    def __init__(self, config) -> None:
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super().__init__()

        self.globals = config.globals
        self.config = config.model

        self.input_embedder = SeqEmbedder(
            **self.config["seq_embedder"]
        )

        self.ss_embedder = SSEmbedder(
            **self.config["ss_embedder"]
        )

        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
            with_x=True
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
            output_s=True
        )
        
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        
        self.heads = End2EndHeads(
            self.config["e2e_heads"],
        )
    
    def iteration(
        self, 
        feats: Dict[str, torch.Tensor], 
        prevs: List[torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        prevs: [m_1_prev, z_prev, x_prev]
        Returns:
            outputs: Dict[str, torch.Tensor]
                msa: 
                    [*, N_seq, N_res, C_m]
                pair: 
                    [*, N_res, N_res, C_z]
                single: 
                    [*, N_res, C_s]
                sm:
                    frames: 
                        [no_blocks, *, N_res, 4, 4]
                    positions: C, P, N
                        [no_blocks, *, N_res, rna_backbone_atom_num, 3]
        """
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)
        
        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        n_res = feats["target_feat"].shape[-2]

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize the MSA and pair representations
        # m: [*, N_seq, N_res, C_m] N_seq=1 for no msa mode
        # z: [*, N_res, N_res, C_z]
        m, z = self.input_embedder(
            tf=feats["target_feat"],
            ri=feats["residue_index"],
            msa=feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

        # ss_emb: [*, N_res, N_res, c_z]
        ss_emb = self.ss_embedder(
            ss=feats["ss"],
        )
        # [*, N_res, N_res, C_z]
        z = add(z, ss_emb, inplace=inplace_safe)

        del ss_emb

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        # Initialize the recycling embeddings, if needs be 
        if m_1_prev is None: 
            # [*, N_res, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n_res, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N_res, N_res, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n_res, n_res, self.config.input_embedder.c_z),
                requires_grad=False,
            )
            
            # [*, N_res, rna_backbone_atom_num, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n_res, nc.rna_backbone_atom_num, 3),
                requires_grad=False,
            )

        # [*, N_res, 3] predicted N coordinates
        glycos_N_x_prev = backbone_atom_fn(
            atom_name="glycos_N", 
            all_atom_positions=x_prev, 
            all_atom_mask=None
        ).to(dtype=z.dtype)

        del x_prev

        # The recycling embedder is memory-intensive, so we offload first
        if self.globals.offload_inference and inplace_safe:
            m = m.cpu()
            z = z.cpu()
        
        # m_1_prev_emb: [*, N_res, C_m]
        # z_prev_emb: [*, N_res, N_res, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m=m_1_prev,
            z=z_prev,
            x=glycos_N_x_prev,
            inplace_safe=inplace_safe,
        )

        del glycos_N_x_prev

        if self.globals.offload_inference and inplace_safe:
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, N_seq, N_res, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N_res, N_res, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, N_seq, N_res, C_m] N_seq=1 for no msa mode
        # z: [*, N_res, N_res, C_z]
        # s: [*, N_res, C_s]
        if self.globals.offload_inference:
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )

            del input_tensors
        else:
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m
        outputs["pair"] = z
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            evoformer_output_dict=outputs,
            restype=feats["restype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        # [*, N_res, rna_backbone_atom_num, 3]
        # outputs["final_atom_positions"] = outputs["sm"]["positions"][-1]
        # outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N_res, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N_res, N_res, C_z]
        z_prev = outputs["pair"]

        # [*, N_res, rna_backbone_atom_num, 3]
        x_prev = outputs["sm"]["positions"][-1]

        return outputs, m_1_prev, z_prev, x_prev

    def _disable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = (
            self.config.evoformer_stack.blocks_per_ckpt
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.
            outputs:
                msa: 
                    [*, N_seq, N_res, C_m]
                pair: 
                    [*, N_res, N_res, C_z]
                single: 
                    [*, N_res, C_s] if not geom mode, else None
                sm: if not geom mode
                    frames: 
                        [no_blocks, *, N_res, 4, 4]
                    positions: C, P, N
                        [no_blocks, *, N_res, rna_backbone_atom_num, 3]
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["restype"].shape[-1]
        num_recycles = 0
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    prevs,
                )

                if not is_final_iter:
                    num_recycles += 1

                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
                else:
                    break

        # Run auxiliary heads
        outputs.update(self.heads(outputs))

        return outputs
