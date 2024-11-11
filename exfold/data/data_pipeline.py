from typing import Dict, Sequence, Optional
import numpy as np
import os

from exfold.data import parsers, mmcif_parsing
from exfold.data.tools import rnafold, petfold
from exfold.data.tools.utils import SSPredictor
from exfold.common import nucleic_constants as nc

FeatureDict = Dict[str, np.ndarray]
SS_PARSERS = {
    "rnafold": parsers.parse_rnafold,
    "petfold": parsers.parse_petfold
}


# region: sequence feats
def make_sequence_features(sequence: str) -> FeatureDict:
    """
    Construct a feature dict of sequence features.
    Args:
        sequence: ACGUCGX
    Returns:
        restype: [N_res, 4+1]
        residue_index: [num_res]
        seq_length: [num_res]
    """
    features = {}
    num_res = len(sequence)
    features["restype"] = nc.sequence_to_onehot(
        sequence=sequence,
        mapping=nc.rna_restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["residue_index"] = np.arange(num_res, dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    return features
# endregion

# region: mmcif feats
def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject, 
    chain_id: str, 
    ptype: str = "rna"
) -> FeatureDict:
    """
    Returns:
        restype: [N_res, 4+1]
        residue_index: [num_res,]
        seq_length: [num_res,]
        all_atom_positions: [N_res, rna_backbone_atom_num, 3]
        all_atom_mask: [N_res, rna_backbone_atom_num]
    """
    mmcif_feats = {}

    input_sequence = mmcif_object.chain_to_seqres[ptype][chain_id]
    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence
        )
    )
    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id, ptype=ptype
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)
    
    return mmcif_feats
# endregion

# region: msa feats
# Generate 1-sequence MSA features having only the input sequence
def make_dummy_msa_feats(input_sequence: str) -> FeatureDict:
    """
    Constructs a feature dict of MSA features.
    Args: 
        input_sequence: ACGUCGX
    Returns:
        msa: [1, N_res] no gap since no msa
    """
    features = {}
    int_msa = []
    int_msa.append([
        nc.rna_restype_order_with_x.get(res, nc.rna_restype_order_with_x["X"]) 
        for res in input_sequence
    ]) # no gap
    features["msa"] = np.array(int_msa, dtype=np.int32)
    return features
# endregion

# region: ss feats
def make_ss_features(ss_data: Sequence[parsers.SS]) -> FeatureDict:
    """
    Constructs a feature dict of ss features.
    Args:
        ss_data: List of SS objects
    Returns:
        ss: [N_res, N_res, 2*len(ss_data)]
    """
    ss_features = []
    
    for ss in ss_data:
        ss_feature = np.stack((ss.pair, ss.prob), axis=-1, dtype=np.float32)
        ss_features.append(ss_feature)

    ss_features = np.concatenate(ss_features, axis=-1, dtype=np.float32)
    return {"ss": ss_features}
# endregion

# todo lm embedding
class DataPipeline:
    """Assembles input features."""
    def __init__(self):
        self.ss_parsers = SS_PARSERS

    def _parse_ss_data(
        self,
        ss_dir: str,
        methods_order: Sequence[str]
    ) -> Sequence[parsers.SS]:
        """
        Parses ss data from .dbn and .prob files for the specified methods.
        Returns a sequence of parsed SS objects in order.
        Args:
            ss_dir: Directory containing .dbn and .prob files.
            methods_order: List of ss methods.
        Returns:
            A sequence of parsed SS objects.
        """
        ss_data = []
        for method in methods_order:
            if method not in self.ss_parsers:
                raise ValueError(f"No parser defined for method {method}.")
            parse_fn = self.ss_parsers[method]
            
            dbn_path = os.path.join(ss_dir, f"{method}.dbn")
            prob_path = os.path.join(ss_dir, f"{method}.prob")
            if not os.path.exists(dbn_path):
                raise FileNotFoundError(f".dbn file for method {method} not found: {dbn_path}")
            if not os.path.exists(prob_path):
                raise FileNotFoundError(f".prob file for method {method} not found: {prob_path}")
            
            with open(dbn_path, "r") as f:
                dbn_string = f.read()
            with open(prob_path, "r") as f:
                prob_string = f.read()

            ss_obj = parse_fn(dbn_string=dbn_string, prob_string=prob_string)
            ss_data.append(ss_obj)
        
        return ss_data

    def _process_ss_feats(
        self, 
        ss_dir: str,
        methods_order: Sequence[str]
    ) -> FeatureDict:
        ss_data = self._parse_ss_data(ss_dir, methods_order)
        ss_features = make_ss_features(ss_data)

        return ss_features

    def process_fasta(
        self,
        fasta_path: str,
        ss_dir: str,
        ss_methods: Sequence[str],
        lm_emb_dir: Optional[str] = None,
        lm_emb_mode: bool = False
    ) -> FeatureDict:
        """
        Assembles features for a single sequence in a FASTA file.
        Returns:
            sequence_features:
                restype: [N_res, 4+1]
                residue_index: [num_res,]
                seq_length: [num_res,]
            msa_features:
                msa: [1, N_res]
            ss_features:
                ss: [N_res, N_res, 2*N_method]
        """
        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, _ = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f"More than one input sequence found in {fasta_path}."
            )
        input_sequence = input_seqs[0]
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
        )

        # If no_msa_mode, generate a dummy MSA features using just the sequence
        msa_features = make_dummy_msa_feats(input_sequence)

        ss_feats = self._process_ss_feats(ss_dir, ss_methods)
        assert len(list(ss_feats.values())[0]) == num_res

        return {
            **sequence_features,
            **msa_features, 
            **ss_feats
        }
    
    def process_mmcif(
        self,
        mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
        chain_id: str,
        ss_dir: str,
        ss_methods: Sequence[str],
        ptype: bool = "rna"
    ) -> FeatureDict:
        """
        Assembles features for a specific chain in an mmCIF object.
        Returns:
            mmcif_features:
                restype: [N_res, 4+1]
                residue_index: [num_res,]
                seq_length: [num_res,]
                all_atom_positions: [N_res, rna_backbone_atom_num, 3]
                all_atom_mask: [N_res, rna_backbone_atom_num]
            msa_features:
                msa: [1, N_res] no gap since no msa
            ss_features:
                ss: [N_res, N_res, 2*N_method]
        """
        if chain_id not in mmcif.structure[0]:
            raise ValueError(f"Chain ID '{chain_id}' not found in {mmcif.file_id}.")
        
        mmcif_features = make_mmcif_features(mmcif, chain_id, ptype)

        input_sequence = mmcif.chain_to_seqres[ptype][chain_id]
        num_res = len(input_sequence)
        # If no_msa_mode, generate a dummy MSA features using just the sequence
        msa_features = make_dummy_msa_feats(input_sequence)

        ss_feats = self._process_ss_feats(ss_dir, ss_methods)
        assert len(list(ss_feats.values())[0]) == num_res

        return {
            **mmcif_features,
            **msa_features,
            **ss_feats
        }


def run_ss_tool(
    ss_runner: SSPredictor,
    fasta_path: str,
    ss_out_prefix: str
) -> Dict[str, str]:
    """
    {ss_out_prefix}.dbn
    {ss_out_prefix}.prob
    """
    result = ss_runner.predict(fasta_path)
    for fmt in result:
        with open(f"{ss_out_prefix}.{fmt}", "w") as f:
            f.write(result[fmt])


class SSRunner:
    def __init__(
        self,
        rnafold_binary_path: Optional[str] = None,
        petfold_binary_path: Optional[str] = None,
    ):
        self.rnafold_runner = None
        if rnafold_binary_path is not None:
            self.rnafold_runner = rnafold.RNAfold(
                binary_path=rnafold_binary_path
            )
        
        self.petfold_runner = None
        if petfold_binary_path is not None:
            self.petfold_runner = petfold.PETfold(
                binary_path=petfold_binary_path
            )
    
    def run(
        self,
        fasta_path: str,
        output_dir: str,
    ):
        """
        Run SS Prediction tools on one sequence.
        可能有多个工具, 比如rnafold和petfold, 结果都保存在output_dir
        """
        if self.rnafold_runner is not None:
            rnafold_out_prefix = os.path.join(output_dir, "rnafold")
            run_ss_tool(
                ss_runner=self.rnafold_runner,
                fasta_path=fasta_path,
                ss_out_prefix=rnafold_out_prefix,
            )
        
        if self.petfold_runner is not None:
            petfold_out_prefix = os.path.join(output_dir, "petfold")
            run_ss_tool(
                ss_runner=self.petfold_runner,
                fasta_path=fasta_path,
                ss_out_prefix=petfold_out_prefix,
            )
