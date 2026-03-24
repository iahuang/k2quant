import os as _os

# Prevent OpenMP thread explosion when using ThreadPoolExecutor with faiss.
# faiss CPU k-means uses OpenMP internally; running it inside a thread pool
# without this causes O(n_threads^2) threads.
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
