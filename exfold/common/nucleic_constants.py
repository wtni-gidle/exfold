import numpy as np
from typing import Dict
# residue
rna_restypes = [
    "A",
    "C",
    "G",
    "U",
]
rna_restype_order = {restype: i for i, restype in enumerate(rna_restypes)}
rna_restype_num = len(rna_restypes)  # := 4.
rna_unk_restype_index = rna_restype_num  # Catch-all index for unknown restypes.

rna_restypes_with_x = rna_restypes + ["X"]
rna_restype_order_with_x = {restype: i for i, restype in enumerate(rna_restypes_with_x)}

rna_restypes_with_x_and_gap = rna_restypes + ["X", "-"]
rna_restype_order_with_x_and_gap =  {restype: i for i, restype in enumerate(rna_restypes_with_x_and_gap)}


def sequence_to_onehot(
    sequence: str, 
    mapping: Dict[str, int], 
    map_unknown_to_x: bool = False
) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
        sequence: 
            An amino acid sequence.
        mapping: 
            A dictionary mapping amino acids to integers.
        map_unknown_to_x: 
            If True, any amino acid that is not in the mapping will be 
            mapped to the unknown amino acid 'X'. If the mapping doesn't contain
            amino acid 'X', an error will be thrown. If False, any amino acid not in 
            the mapping will throw an error.

    Returns:
        A numpy array of shape (seq_len, num_) with one-hot encoding of
        the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 "
            "without any gaps. Got: %s" % sorted(mapping.values())
        )

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for idx, res_type in enumerate(sequence):
        if map_unknown_to_x:
            if res_type.isalpha() and res_type.isupper():
                res_id = mapping.get(res_type, mapping["X"])
            else:
                raise ValueError(
                    f"Invalid character in the sequence: {res_type}"
                )
        else:
            res_id = mapping[res_type]
        one_hot_arr[idx, res_id] = 1

    return one_hot_arr

# atom

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
rna_atoms = [
    "OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
    "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", 
    "N3", "C4", "O2", "N4", "O6", "N2", "O4"
]
rna_atom_order = {atom: i for i, atom in enumerate(rna_atoms)}
rna_atom_num = len(rna_atoms)  # := 28.

# residue and atom
# A list of atoms (excluding hydrogen) for each nucleotide type. 
# https://www.ebi.ac.uk/pdbe-srv/pdbechem
rna_residue_atoms = {
    "A": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
          "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", 
          "N3", "C4"],
    "C": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "G": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2",
          "N2", "N3", "C4"],
    "U": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
          "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}

# Atom positions relative to backbone.
# format: {residue_type: {atom_name: (x, y, z)}}
rna_backbone_atom_positions = {
    "A": {
        "P": (2.3240912, -0.80317456, -1.5209152),
        "C4'": (-0.8031747, 1.3708649, -0.56768906),
        "N9": (-1.520915, -0.5676891, 2.088603),
    },
    "C": {
        "P": (2.35021, -0.7880346, -1.5621773),
        "C4'": (-0.78803474, 1.3541472, -0.566115),
        "N1": (-1.5621774, -0.56611556, 2.1282897),
    },
    "G": {
        "P": (2.3250363, -0.80280226, -1.5222337),
        "C4'": (-0.8028024, 1.3702725, -0.5674678),
        "N9": (-1.5222336, -0.5674678, 2.0897026),
    },
    "U": {
        "P": (2.3493478, -0.7881313, -1.5612125),
        "C4'": (-0.7881318, 1.3542684, -0.5661326),
        "N1": (-1.5612122, -0.5661328, 2.1273444),
    },
}

rna_backbone_atoms = {
    "A": ["P", "C4'", "N9"],
    "C": ["P", "C4'", "N1"],
    "G": ["P", "C4'", "N9"],
    "U": ["P", "C4'", "N1"],
}
rna_backbone_atom_num = len(rna_backbone_atoms["A"])  # := 3.

rna_restype_atom3_backbone_positions = np.zeros(
    [rna_restype_num + 1, rna_backbone_atom_num, 3], # ACGUX, 3 atoms, xyz
    dtype=np.float32
)
for restype, res_idx in rna_restype_order.items():
    for atomname, atom_position in rna_backbone_atom_positions[restype].items():
        atom_idx = rna_backbone_atoms[restype].index(atomname)
        rna_restype_atom3_backbone_positions[res_idx, atom_idx, :] = atom_position
