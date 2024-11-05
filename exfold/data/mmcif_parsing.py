from typing import Any, Optional, Sequence, Tuple, Generator, Dict
import io
import logging
from functools import partial
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from Bio import PDB
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Residue import DisorderedResidue
from Bio.Data import PDBData

from exfold.common import nucleic_constants as nc
from exfold.data.errors import MultipleChainsError


# Type aliases:
ChainId = str
PdbHeader = Dict[str, Any]
PdbStructure = Structure
PdbModel = Model
SeqRes = str
MmCIFDict = Dict[str, Sequence[str]]
PType = str

PTYPE_LOOKUP: Dict[PType, str] = {
    "protein": "peptide",
    "rna": "RNA",
    "dna": "DNA"
}

CHAIN_IDS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

@dataclass(frozen=True)
class Monomer:
    """
    id: _entity_poly_seq.mon_id
    num: _entity_poly_seq.num
    """
    id: str
    num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: str
    insertion_code: str
    hetatm_atom: str
    model_num: str


# Used to map SEQRES index to a residue in the structure.
@dataclass(frozen=True)
class ResidueID:
    hetflag: str
    number: int
    insertion_code: str


@dataclass(frozen=True)
class ResidueAtPosition:
    chain_id: str
    residue_id: Optional[ResidueID]
    name: str
    is_missing: bool


@dataclass(frozen=True)
class MmcifObject:
    """
    Representation of a parsed mmCIF file.
        file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all files being processed.
        header: Header extracted from _mmcif_dict
        structure: The structure object modified from Biopython structure.
        chain_to_seqres: Dict mapping custom_chain_id to one letter sequence. E.g.
        {'A': 'ABCDEFG'}
        seqres_to_structure: Dict mapping custom_chain_id to a dictionary that connects SEQRES indices 
                      to ResidueAtPosition objects. 
                      Example: {'A': {0: ResidueAtPosition, 1: ResidueAtPosition, ...}}
        mmcif_to_custom_mapping: Dict mapping mmcif chain IDs to custom chain IDs.
        raw_string: The raw string used to construct the MmcifObject.
    """
    file_id: str
    header: PdbHeader
    structure: PdbStructure #* 注意是Structure而不是Model，需要structure[0]
    chain_to_seqres: Dict[PType, Dict[ChainId, SeqRes]]
    seqres_to_structure: Dict[ChainId, Dict[int, ResidueAtPosition]]
    mmcif_to_custom_mapping: Dict[ChainId, ChainId]
    raw_string: Any


@dataclass(frozen=True)
class ParsingResult:
    """
    Returned by the parse function.
        mmcif_object: A MmcifObject, may be None if no chain could be successfully parsed.
        errors: A dict mapping (file_id, chain_id) to any exception generated.
    """
    mmcif_object: Optional[MmcifObject]
    errors: Dict[Tuple[str, str], Any]


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


def parse(
    file_id: str, 
    mmcif_string: str, 
    catch_all_errors: bool = True
) -> ParsingResult:
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        structure = parser.get_structure("", handle)
        structure = _get_first_model(structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info, file_id)
        # Get all valid chains
        valid_chains = _get_all_valid_chains(parsed_info)

        # Check if there are no valid protein or nucleic acid chains
        if not any(valid_chains.values()):
            return ParsingResult(
                None, {(file_id, ""): ParseError(f"No protein or nucleic acid chains found in {file_id}.")}
            )
        # Identify hybrid chains
        chain_ids = [chain_id for inner_dict in valid_chains.values() for chain_id in inner_dict.keys()]
        if len(chain_ids) != len(set(chain_ids)):
            return ParsingResult(
                None, {(file_id, ""): ParseError(f"Hybrid chains found in {file_id}.")}
            )
                
        # Guarantee all chains are included in ATOM records
        mmcif_chain_ids_in_record = parsed_info["_atom_site.label_asym_id"]
        for ptype, chains in valid_chains.items():
            for chain_id in list(chains.keys()):
                if chain_id not in mmcif_chain_ids_in_record:
                    return ParsingResult(
                        None, {(file_id, ""): ParseError(f"Some chains not found in {file_id}.")}
                    )

        flat_valid_chains: Dict[ChainId, Sequence[Monomer]] = {}
        for chains in valid_chains.values():
            flat_valid_chains.update(chains)
        
        # Check parsed sequences are consecutive increasing sequences starting from 1
        for chain_id, seq in flat_valid_chains.items():
            intl_nums = [monomer.num for monomer in seq]
            if intl_nums != list(range(1, len(intl_nums) + 1)):
                return ParsingResult(
                    None, {(file_id, chain_id): ParseError(
                        "The parsed sequence is not a consecutive increasing sequence "
                        f"starting from 1, which is not expected: {file_id}_{chain_id}.")}
                )
        
        # Create a structure corresponding to our parsed results, 
        # modifying chain names for saving as a PDB file if needed.
        new_structure = PDB.Structure.Structure('structure')
        new_model = PDB.Model.Model(0)
        new_structure.add(new_model)

        mmcif_to_custom_chain_id = dict(zip(flat_valid_chains.keys(), CHAIN_IDS))
        if len(flat_valid_chains) > len(CHAIN_IDS):
            mmcif_to_custom_chain_id = dict(zip(flat_valid_chains.keys(), flat_valid_chains.keys()))
            logging.info(f"Too many chains in the mmCIF file, we had to use default chain IDs: {file_id}")

        for chain_id in mmcif_to_custom_chain_id.values():
            new_chain = PDB.Chain.Chain(chain_id)
            new_model.add(new_chain)
        
        # region: seq_to_structure_mappings
        seq_to_structure_mappings: Dict[ChainId, Dict[int, ResidueAtPosition]] = defaultdict(dict)
        min_model_num = min([int(num) for num in parsed_info["_atom_site.pdbx_PDB_model_num"]])
        for atom in _get_atom_site_generator(parsed_info):
            # We only process the first model.
            if int(atom.model_num) != min_model_num:
                continue

            # We only process the atoms of interest.
            if atom.mmcif_chain_id not in flat_valid_chains.keys():
                continue

            # region: seq_idx
            seq_idx = int(atom.mmcif_seq_num) - 1
            # endregion
            # custom_chain_id
            custom_chain_id = mmcif_to_custom_chain_id[atom.mmcif_chain_id]

            # 如果seq_idx已经存在，即当前残基已被记录，则无需重复记录
            if seq_idx not in seq_to_structure_mappings[custom_chain_id].keys():
                # region: hetflag
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                # endregion
                # region: insertion_code
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                # endregion
                res_name = flat_valid_chains[atom.mmcif_chain_id][seq_idx].id
                residue_id = (hetflag, int(atom.author_seq_num), insertion_code)

                # 如果是alternative，选择valid_chains中的氨基酸
                residue = structure[atom.author_chain_id][residue_id]
                if isinstance(residue, DisorderedResidue):
                    residue.disordered_select(res_name)
                
                new_residue = residue.copy()
                new_residue.id = (hetflag, int(atom.mmcif_seq_num), insertion_code)
                new_model[custom_chain_id].add(new_residue)

                # region: current
                current = seq_to_structure_mappings[custom_chain_id]
                current[seq_idx] = ResidueAtPosition(
                    chain_id=custom_chain_id,
                    residue_id=ResidueID(*new_residue.id),
                    name=res_name,
                    is_missing=False
                )
                seq_to_structure_mappings[custom_chain_id] = current
                # endregion
        #* check altloc, crucial for complex datasets.
        altloc_is_ok = _check_altloc(new_model)
        if not altloc_is_ok:
            logging.warning("Potential issues with chain alternative conformations; "
                            f"check if focusing on complex datasets: {file_id}.")
        
        # Add missing residue information to seq_to_structure_mappings.
        for ptype, chains in valid_chains.items():
            for mmcif_chain_id, seq_info in chains.items():
                custom_chain_id = mmcif_to_custom_chain_id[mmcif_chain_id]
                current_mapping = seq_to_structure_mappings[custom_chain_id]
                for idx, monomer in enumerate(seq_info):
                    if idx not in current_mapping:
                        current_mapping[idx] = ResidueAtPosition(
                            chain_id=custom_chain_id,
                            residue_id=None,
                            name=monomer.id,
                            is_missing=True
                        )
        # endregion
        # region: author_chain_to_sequence
        custom_chain_to_sequence: Dict[PType, Dict[ChainId, SeqRes]] = {
            ptype: dict()
            for ptype in PTYPE_LOOKUP.keys()
        }
        for ptype, chains in valid_chains.items():
            for chain_id, monomers in chains.items():
                custom_chain_id = mmcif_to_custom_chain_id[chain_id]
                sequence = _letters_3to1(monomers, ptype)
                custom_chain_to_sequence[ptype][custom_chain_id] = sequence
        # endregion

        # structure and raw_string occupy the majority of the memory, each accounting for about half.
        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=new_structure,
            chain_to_seqres=custom_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            mmcif_to_custom_mapping=mmcif_to_custom_chain_id, 
            raw_string=parsed_info,
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise 
        return ParsingResult(mmcif_object=None, errors=errors)

    
def _get_first_model(structure: PdbStructure) -> PdbModel:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())

# region header

def _get_release_date(parsed_info: MmCIFDict, file_id: str) -> str:
    """Returns the oldest revision date."""
    release_date = "?"
    if "_pdbx_audit_revision_history.revision_date" in parsed_info.keys():
        release_date = min(parsed_info["_pdbx_audit_revision_history.revision_date"])
    else:
        logging.warning(
            "Could not determine release_date in %s", file_id
        )
    
    return release_date

def _get_resolution(parsed_info: MmCIFDict, file_id: str) -> float:
    resolution = 0.00
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                resolution = float(raw_resolution)
                break
            except ValueError:
                logging.info(
                    "Invalid resolution format in %s", file_id
                )

    return resolution

def _get_header(parsed_info: MmCIFDict, file_id: str) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    # structure_method: 实验方法
    header["structure_method"] = ",".join(parsed_info["_exptl.method"]).lower()
    # release_date: 最早的revision日期，默认为"?"
    header["release_date"] = _get_release_date(parsed_info, file_id)
    # resolution: 分辨率, 三种数据项按顺序择其一; 默认为0.0
    header["resolution"] = _get_resolution(parsed_info, file_id)

    return header

# endregion

def _get_all_valid_chains(
    parsed_info: Dict[str, Any]
) -> Dict[PType, Dict[ChainId, Sequence[Monomer]]]:
    """Extracts polymer information for protein/RNA/DNA chains only.

    Args:
      parsed_info: _mmcif_dict produced by the Biopython parser.

    Returns:
      A dict mapping mmcif chain id to a list of Monomers.
    """
    # region: polymers  
    # Get polymer information. Polymer is a unique chain sequence of proteins, nucleic acids.
    ent_ids = parsed_info["_entity_poly_seq.entity_id"]
    mon_ids = parsed_info["_entity_poly_seq.mon_id"]
    intl_nums = parsed_info["_entity_poly_seq.num"]

    polymers = defaultdict(list)
    # To make the parsed sequence as close as possible to one_letter_code_can, 
    # we always select the first for residues with alternative conformations.
    seen_nums = []
    for ent_id, mon_id, intl_num in zip(ent_ids, mon_ids, intl_nums):
        if (ent_id, intl_num) not in seen_nums:
            polymers[ent_id].append(Monomer(id=mon_id, num=int(intl_num)))
            seen_nums.append((ent_id, intl_num))
    # endregion
    
    # region: chem_comps  
    # Get chemical compositions. Will allow us to identify types of polymers.
    chem_comps = dict(zip(parsed_info["_chem_comp.id"], parsed_info["_chem_comp.type"]))
    # endregion

    # region: entity_to_mmcif_chains
    # Get chains information for each polymer. Necessary so that we can return a 
    # dict keyed on chain id rather than entity.
    entity_to_mmcif_chains = defaultdict(list)
    for chain_id, ent_id in zip(parsed_info["_struct_asym.id"], parsed_info["_struct_asym.entity_id"]):
        entity_to_mmcif_chains[ent_id].append(chain_id)
    # endregion
    
    # region: valid_chains
    # Identify and return the chains containing specific residues.
    get_valid_chains = partial(_get_valid_chains, polymers, chem_comps, entity_to_mmcif_chains)

    valid_chains = {}
    for ptype, term in PTYPE_LOOKUP.items():
        valid_chains[ptype] = get_valid_chains(term)
    # endregion

    return valid_chains

def _get_valid_chains(
    polymers: Dict[str, Sequence[Monomer]], 
    chem_comps: Dict[str, str], 
    entity_to_mmcif_chains: Dict[str, list[ChainId]],
    term: str
) -> Dict[ChainId, Sequence[Monomer]]:
    """
    term: "peptide", "RNA", "DNA"
    """
    valid_chains = {}
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]
        if any(
            [
                term in chem_comps[monomer.id]
                for monomer in seq_info
            ]
        ):
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info
    return valid_chains

def _get_atom_site_generator(parsed_info: MmCIFDict) -> Generator[AtomSite, None, None]:
    """Returns list of atom sites; contains data not present in the structure."""
    return (
        AtomSite(*site)
        for site in zip(  # pylint:disable=g-complex-comprehension
            parsed_info["_atom_site.label_comp_id"],
            parsed_info["_atom_site.auth_asym_id"],
            parsed_info["_atom_site.label_asym_id"],
            parsed_info["_atom_site.auth_seq_id"],
            parsed_info["_atom_site.label_seq_id"],
            parsed_info["_atom_site.pdbx_PDB_ins_code"],
            parsed_info["_atom_site.group_PDB"],
            parsed_info["_atom_site.pdbx_PDB_model_num"]
        )
    )

def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in (".", "?")

def _letters_3to1(monomers: Sequence[Monomer], ptype: PType):
    """The obtained sequence may include are non-standard characters."""
    if ptype == "protein":
        letters_3to1_dict = PDBData.protein_letters_3to1_extended
    else:
        letters_3to1_dict = PDBData.nucleic_letters_3to1_extended
    seq = []
    for monomer in monomers:
        # pad to 3 characters
        letter = monomer.id + " " * (3 - len(monomer.id))
        code = letters_3to1_dict.get(letter, "X")
        seq.append(code if len(code) == 1 else "X")
    seq = "".join(seq)
    
    return seq


# def compare_monomers_seq_code(monomers, seq_code, ptype: PType):
#     monomers_one_letter = _letters_3to1(monomers, ptype)
#     # 使用正则表达式匹配括号内的内容或单个字符
#     pattern = r'\([^)]*\)|.'
#     # 使用 findall 提取匹配项，并去除括号
#     result = re.findall(pattern, seq_code)
#     # 去除括号并返回处理后的列表
#     seq_code = [item[1:-1] if item.startswith('(') else item for item in result]
#     if len(seq_code) != len(monomers_one_letter):
#         length_eq = False
#     else:
#         length_eq = True
#     def all_equal_to_target(list1, list2, target):
#         if len(list1) != len(target) or len(list2) != len(target):
#             return False
#         for a, b, t in zip(list1, list2, target):
#             if a != t and b != t:
#                 print(a, b, t, flush=True)
#         return all(a == t or b == t for a, b, t in zip(list1, list2, target))
#     right = all_equal_to_target([monomer.id for monomer in monomers], list(monomers_one_letter), seq_code)
    
#     return length_eq, right, monomers_one_letter


def _check_altloc(model):
    """
    This function identifies alternative conformations for entire chains in PDB files.

    Handling alternative conformations can be complex due to issues such as:
    - Scene 1: Multiple models leading to different conformations.
    - Scene 2: Short chains with occupancy less than 1.0.
    - Scene 3: Two chains representing the same underlying chain with different chain ids (e.g., in 8eb5).
    - etc.
    
    For monomer (single chain) datasets, users can ignore this function and the warnings it generates.
    For complex datasets, user should note that multiple chains may be alternative forms of one chain (Scene 3).
    - Check the entries identified by this function to decide how to handle them, instead of deleting them directly.
    - To reduce workload, fist mark the identified entries and check them after completing general dataset filtering.
    """
    for chain in model:
        residue_occupancy_list = []
        for residue in chain:
            if not residue.is_disordered():
                residue_occupancy_list.append(1.0)
                continue
            if isinstance(residue, DisorderedResidue):
                residues = residue.disordered_get_list()
            else:
                residues = [residue]
            res_occupancy_list = []
            for res in residues:
                atom_occupancy_list = []
                for atom in res:
                    if atom.is_disordered():
                        atoms = atom.disordered_get_list()
                        atom_occupancy = sum(a.occupancy for a in atoms)
                    else:
                        atom_occupancy = atom.occupancy
                    atom_occupancy_list.append(atom_occupancy)
                res_occupancy = np.mean(atom_occupancy_list)
                if 1.0 in atom_occupancy_list:
                    res_occupancy = 1.0
                res_occupancy_list.append(res_occupancy)
            residue_occupancy = np.sum(res_occupancy_list)
            residue_occupancy_list.append(residue_occupancy)
        # if np.mean(np.array(residue_occupancy_list) > 0.8) < 0.9:
        #     return False
        if all(np.array(residue_occupancy_list) < 1.0):
            return False

    return True


def get_atom_coords(
    mmcif_object: MmcifObject,
    chain_id: str,
    ptype: PType
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get atom coordinates and mask from mmcif object.
    只有残基为标准残基且被记录的才会有坐标(mask=1). 序列使用chain_to_seqres
    其他情况下原子的mask为0.
    Returns:
        all_atom_positions: [N_res, rna_backbone_atom_num, 3]
        all_atom_mask: [N_res, rna_backbone_atom_num]
    """
    # Locate the right chain
    chains = list(mmcif_object.structure.get_chains())
    relevant_chains = [c for c in chains if c.id == chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(
            f"Expected exactly one chain in structure with id '{chain_id}'."
        )
    chain = relevant_chains[0]

    # Extract the coordinates
    sequence = mmcif_object.chain_to_seqres[ptype][chain_id]
    num_res = len(sequence)
    all_atom_positions = np.zeros(
        [num_res, nc.rna_backbone_atom_num, 3], dtype=np.float32 # [N_res, 3 bb atoms, 3]
    )
    all_atom_mask = np.zeros(
        [num_res, nc.rna_backbone_atom_num], dtype=np.float32 # [N_res, 3 bb atoms]
    )
    for res_index in range(num_res):
        position = np.zeros([nc.rna_backbone_atom_num, 3], dtype=np.float32)
        mask = np.zeros([nc.rna_backbone_atom_num], dtype=np.float32)
        res_name = sequence[res_index]
        # 如果res_name是标准残基ACGU中的一者, 则提取坐标
        if res_name in nc.rna_backbone_atoms:
            bb_atoms = nc.rna_backbone_atoms[res_name]
            res_at_position = mmcif_object.seqres_to_structure[chain_id][res_index]
            if not res_at_position.is_missing:
                res = chain[
                    (
                        res_at_position.residue_id.hetflag,
                        res_at_position.residue_id.number,
                        res_at_position.residue_id.insertion_code,
                    )
                ]
                for atom in res.get_atoms():
                    atom_name = atom.get_name()
                    x, y, z = atom.get_coord()
                    if atom_name in bb_atoms:
                        position[bb_atoms.index(atom_name)] = [x, y, z]
                        mask[bb_atoms.index(atom_name)] = 1.0
        all_atom_positions[res_index] = position
        all_atom_mask[res_index] = mask
    
    return all_atom_positions, all_atom_mask

