import bz2
import gzip
import lzma
from pathlib import Path

import numpy as np
from src.problems.utils.data import Data


def _open_libsvm(path):
    """
    Open a LIBSVM file that might be plain text, gzip, or bzip2.
    """
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(6)
    if magic.startswith(b"\x1f\x8b"):  # gzip magic
        return gzip.open(path, "rt", encoding="latin-1", errors="ignore")
    if magic.startswith(b"BZh"):  # bzip2 magic
        return bz2.open(path, "rt", encoding="latin-1", errors="ignore")
    if magic.startswith(b"\xfd7zXZ\x00"):  # xz magic
        return lzma.open(path, "rt", encoding="latin-1", errors="ignore")
    return path.open("r", encoding="latin-1", errors="ignore")


def libsvm_parser(path, n, d):
    features = np.zeros((n, d))
    values = np.zeros(n)

    # Auto-handle compressed inputs and tolerate non-UTF8 bytes.
    with _open_libsvm(path) as f:
        try:
            data_str = f.readlines()
        except EOFError:
            # Handle truncated compressed files gracefully; keep what we could read.
            f.seek(0)
            data_str = []
            try:
                for line in f:
                    data_str.append(line)
            except EOFError:
                pass

    for i in range(min(n, len(data_str))):
        parts = data_str[i].strip().split()
        if not parts:
            continue
        values[i] = float(parts[0])

        for fv in parts[1:]:
            idx, feature = fv.split(":")
            features[i, int(idx) - 1] = float(feature)  # Convert to 0-based index for Python

    return Data(features, values)
