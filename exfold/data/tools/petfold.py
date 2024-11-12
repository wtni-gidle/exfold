"""Library to run PETfold from Python."""
from typing import Dict
import os
import logging
import subprocess
import tempfile
import numpy as np
from scipy.sparse import csr_matrix

from exfold.data.tools import utils


class PETfold(utils.SSPredictor):
    """Python wrapper of the PETfold binary."""
    def __init__(self, binary_path: str):
        self.binary_path = binary_path
    
    #! 即使不报错，也可能出现空文件，所以需要后续检查文件内容
    def predict(self, input_fasta_path: str) -> Dict[str, str]:
        """
        输出的key即format, value必须是字符串
        """
        input_fasta_path = os.path.abspath(input_fasta_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(input_fasta_path, "r") as f:
                input_fasta = f.read()

            input_path = os.path.join(tmp_dir, "input.fasta")
            # petfold need MSA as input
            alignments = input_fasta * 3 if input_fasta.endswith("\n") else (input_fasta + "\n") * 3
            with open(input_path, "w") as f:
                f.write(alignments)

            pp_file = os.path.join(tmp_dir, "petfold.pp")
            
            cmd_flags = [
                "-f", input_path,
                "-r", pp_file,
            ]
            cmd = [self.binary_path] + cmd_flags

            logging.info('Launching subprocess "%s"', " ".join(cmd))
            with utils.timing(f"PETfold predict"):
                result = subprocess.run(cmd, capture_output=True, text=True)
                retcode = result.returncode
                stdout = result.stdout
                stderr = result.stderr
            
            if retcode:
                raise RuntimeError(
                    f"PETfold failed for {input_fasta_path}\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
                )
            
            # extract dot-bracket notation
            dbn = self._extract_DBN(stdout)

            # extract probability matrix lines
            prob = self._extract_prob_mat(pp_file)
        
        raw_output = {
            "dbn": dbn,
            "prob": prob,
        }

        return raw_output
    
    @staticmethod
    def _extract_DBN(stdout: str) -> str:
        """extract dot-bracket notation"""
        for line in stdout.splitlines():
            if line.startswith("PETfold"):
                dbn = line.split(":")[1].strip()
                break

        return dbn
    
    @staticmethod
    def _extract_prob_mat(pp_file: str) -> str:
        """
        extract probability matrix and only write nonzeros.
        This is very inelegant, but idk how else to reduce storage size.
        每一行格式如下：
        i j p
        """
        prob_matrix = np.genfromtxt(pp_file, skip_header=1, skip_footer=1)
        prob_matrix = csr_matrix(prob_matrix)
        prob_str = ""
        for row, col in zip(*prob_matrix.nonzero()):
            if row <= col:
                value = prob_matrix[row, col]
                prob_str += f"{row+1} {col+1} {value}\n"
                
        return prob_str
