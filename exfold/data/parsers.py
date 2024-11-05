"""Functions for parsing various file formats."""
import dataclasses
import numpy as np
from Bio import SeqIO
from io import StringIO
from typing import Tuple, Sequence, Optional


@dataclasses.dataclass(frozen=True)
class SS:
    """
    pair: [N_res, N_res] whether the residue pair forms a base pair or not
    prob: [N_res, N_res] probability map
    """
    pair: Optional[np.ndarray]
    prob: Optional[np.ndarray]

    def __post_init__(self):
        if not self.pair.shape == self.prob.shape:
            raise ValueError(
                "All fields for SS must have the same shape."
            )


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    fasta_io = StringIO(fasta_string)
    sequences = []
    descriptions = []

    for record in SeqIO.parse(fasta_io, "fasta"):
        descriptions.append(record.description)
        sequences.append(str(record.seq))

    return sequences, descriptions


def _parse_dbn(dbn_string: str) -> np.ndarray:
    """
    dbn -> [N_res, N_res] 0/1 matrix

    Example:
        >>> dbn_string = "(((.((..(((.))))))..))"
        >>> parse_dbn(dbn_string)
        array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=int32)
    """
    n_res = len(dbn_string)
    ss_matrix = np.zeros((n_res, n_res), dtype=np.int32)
    slist = []

    open_count = sum(1 for char in dbn_string if char in "(<[{")
    close_count = sum(1 for char in dbn_string if char in ")>]}")
    assert open_count == close_count, f'Unbalanced brackets: {dbn_string}'

    for i, char in enumerate(dbn_string):
        if char in "(<[{":
            slist.append(i)
        elif char in ")>]}":
            j = slist.pop()
            ss_matrix[i, j] = ss_matrix[j, i] = 1
        elif char not in  ".-":
            raise ValueError(
                f'Unknown secondary structure state: {char} at position {i}'
            )
        
    return ss_matrix


def parse_rnafold(dbn_string: str, prob_string: str) -> SS:
    # base pair
    ss_matrix = _parse_dbn(dbn_string)

    # probability map
    n_res = len(dbn_string)
    prob_matrix = np.zeros((n_res, n_res), dtype=np.float32)
    prob_string = prob_string.split("\n")
    for line in prob_string:
        if line == "":
            continue
        words = line.split()
        i = int(words[0]) - 1
        j = int(words[1]) - 1
        score = float(words[2])
        prob_matrix[i, j] = prob_matrix[j, i] = score
    
    return SS(
        pair=ss_matrix, 
        prob=prob_matrix
    )

