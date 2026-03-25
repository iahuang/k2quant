import os as _os

# Prevent OpenMP/OpenBLAS thread explosion inside the C++ worker threads.
# The C++ kernel manages its own thread pool; without this, each worker's
# cblas calls may spawn additional OpenMP threads.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

from .vq import vq_quantize, vq_reconstruct, VQResult
from .config import QuantConfig
from .kbvq import bcos, idre

__all__ = [
    "vq_quantize",
    "vq_reconstruct",
    "VQResult",
    "QuantConfig",
    "bcos",
    "idre",
]
