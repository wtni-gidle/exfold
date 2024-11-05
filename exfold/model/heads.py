from typing import Dict

import torch
import torch.nn as nn

from exfold.model.primitives import Linear, LayerNorm
from exfold.utils.precision_utils import is_fp16_enabled


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(
        self, 
        c_z: int, 
        no_bins: int,
        **kwargs
    ):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def _forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, no_bins] distogram probability distribution
        """
        # [*, N_res, N_res, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
    
    def forward(self, z: torch.Tensor) -> torch.Tensor: 
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(z.float())
        else:
            return self._forward(z)


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(
        self, 
        c_m: int, 
        c_out: int, 
        **kwargs
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, init="final")

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        # [*, N_seq, N_res, C_out]
        logits = self.linear(m)
        return logits


class End2EndHeads(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.masked_msa = MaskedMSAHead(
            **config["masked_msa"],
        )

        self.config = config
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        e2e_out = {}

        distogram_logits = self.distogram(outputs["pair"])
        e2e_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        e2e_out["masked_msa_logits"] = masked_msa_logits

        return e2e_out


class ResnetBlock(nn.Module):
    def __init__(self, c_hidden: int):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [*, c_hidden]
        return : [*, c_hidden]
        """
        x_initial = x

        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)

        return x + x_initial


class Geometry1DHead(nn.Module):
    def __init__(
        self, 
        c_s: int, 
        c_hidden: int, 
        no_blocks: int,
        no_bins: int
    ):
        """
        Args:
            c_s:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_bins:
                Number of bins
        """
        super().__init__()

        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_bins = no_bins

        self.linear_in = Linear(self.c_s, self.c_hidden, init="relu")
        self.layer_norm = LayerNorm(self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)
        
        self.linear_out = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def _forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N_res, no_bins] predicted angles
        """
        s = self.linear_in(s)
        s = self.layer_norm(s)
        
        # [*, N_res, C_hidden]
        for l in self.layers:
            s = l(s)
        
        s = self.relu(s)

        # [*, N_res, no_bins]
        logits = self.linear_out(s)

        return logits
    
    def forward(self, s: torch.Tensor) -> torch.Tensor: 
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(s.float())
        else:
            return self._forward(s)


class Geometry2DHead(nn.Module):
    def __init__(
        self, 
        c_z: int, 
        c_hidden: int, 
        no_blocks: int,
        no_bins: int,
        symmetrize: bool = True
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_bins:
                Number of bins
        """
        super().__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_bins = no_bins
        self.symmetrize = symmetrize

        self.linear_in = Linear(self.c_z, self.c_hidden, init="relu")
        self.layer_norm = LayerNorm(self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = ResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def _forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, no_bins] predicted angles
        """
        # [*, N_res, N_res, C_hidden]
        z = self.linear_in(z)
        z = self.layer_norm(z)

        for l in self.layers:
            z = l(z)

        z = self.relu(z)
        # [*, N_res, N_res, no_bins]
        logits = self.linear_out(z)

        if self.symmetrize:
            logits = logits + logits.transpose(-2, -3)

        return logits
    
    def forward(self, z: torch.Tensor) -> torch.Tensor: 
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(z.float())
        else:
            return self._forward(z)


class GeometryHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pp_head = Geometry2DHead(
            **config["PP"]
        )
        self.cc_head = Geometry2DHead(
            **config["CC"]
        )
        self.nn_head = Geometry2DHead(
            **config["NN"]
        )
        self.pccp_head = Geometry2DHead(
            **config["PCCP"]
        )
        self.cnnc_head = Geometry2DHead(
            **config["CNNC"]
        )
        self.pnnp_head = Geometry2DHead(
            **config["PNNP"]
        )
        self.masked_msa = MaskedMSAHead(
            **config["masked_msa"],
        )
        
        self.config = config
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        geom_outputs = {}
        geom_outputs["PP_logits"] = self.pp_head(outputs["pair"])
        geom_outputs["CC_logits"] = self.cc_head(outputs["pair"])
        geom_outputs["NN_logits"] = self.nn_head(outputs["pair"])
        geom_outputs["PCCP_logits"] = self.pccp_head(outputs["pair"])
        geom_outputs["PNNP_logits"] = self.pnnp_head(outputs["pair"])
        geom_outputs["CNNC_logits"] = self.cnnc_head(outputs["pair"])
        geom_outputs["masked_msa_logits"] = self.masked_msa(outputs["msa"])

        return geom_outputs