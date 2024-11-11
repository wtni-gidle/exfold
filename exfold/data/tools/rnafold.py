"""Library to run RNAfold from Python."""
from typing import Dict
import os
import logging
import subprocess
import tempfile

from exfold.data.tools import utils


class RNAfold(utils.SSPredictor):
    """Python wrapper of the RNAfold binary."""

    def __init__(self, binary_path: str):
        self.binary_path = binary_path
    
    #! 即使不报错，也可能出现空文件，所以需要后续检查文件内容
    def predict(self, input_fasta_path: str) -> Dict[str, str]:
        """
        输出的key即format, value必须是字符串
        """
        input_fasta_path = os.path.abspath(input_fasta_path)
        ori_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)

            outfile_path = "rnafold.ss"
            cmd_flags = [
                "--outfile=" + outfile_path,
                "-p",
                "--noPS",
            ]
            cmd = [self.binary_path] + cmd_flags + [input_fasta_path]
            
            logging.info('Launching subprocess "%s"', " ".join(cmd))
            with utils.timing(f"RNAfold predict..."):
                result = subprocess.run(cmd, capture_output=True, text=True)
                retcode = result.returncode
                stdout = result.stdout
                stderr = result.stderr

            if retcode:
                os.chdir(ori_dir)
                raise RuntimeError(
                    f"RNAfold failed for {input_fasta_path}\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
                )
            
            # extract dot-bracket notation
            with open(outfile_path, "r") as f:
                ss_str = f.read()
            dbn = self._extract_DBN(ss_str)

            # extract probability matrix lines
            dp_file = [fn for fn in os.listdir() if fn.endswith("dp.ps")]
            assert len(dp_file) == 1
            with open(dp_file[0], "r") as f:
                dp_str = f.read()
            prob = self._extract_prob_mat(dp_str)
            
            os.chdir(ori_dir)

        raw_output = {
            "dbn": dbn,
            "prob": prob,
        }

        return raw_output
    
    @staticmethod
    def _extract_DBN(ss_str: str) -> str:
        """extract dot-bracket notation"""
        ss_str = ss_str.split("\n")
        dbn = ss_str[2].strip().split()[0]

        return dbn

    @staticmethod
    def _extract_prob_mat(dp_str: str) -> str:
        """
        extract probability matrix lines
        每一行格式如下：
        i j sqrt(p)
        """
        dp_str = dp_str.split("\n")
        dp_str = [line.strip() for line in dp_str]
        prob_mat_lines = []
        for line in dp_str:
            line = line.strip()
            if (line.endswith("ubox") and 
                len(line.split()) == 4 and 
                line[0].isdigit()):
                prob_mat_lines.append(" ".join(line.split()[:-1]))
        prob = "\n".join(prob_mat_lines)

        return prob
