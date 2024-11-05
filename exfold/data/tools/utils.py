"""Common utilities for data pipeline tools."""
from typing import Optional, Dict
import contextlib
import datetime
import logging
import shutil
import tempfile
import time
from abc import ABC, abstractmethod


class SSPredictor(ABC):
    """
    RNA second structure prediction tool base class
    """
    @abstractmethod
    def predict(self) -> Dict[str, str]:
        """
        Returns
            dbn
            prob
        """
        pass


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)


def to_date(s: str):
    return datetime.datetime(
        year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10])
    )

