from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class QuantConfig:
    """All hyperparameters for the IDRE + VQ + BCOS quantization pipeline.

    Attributes:
        k_factor: IDRE truncated SVD rank as fraction of input dim.
            k = int(ic * k_factor). Higher values retain more shared structure
            at full precision, leaving a smaller residual for VQ.
            Paper recommends 1/128, but this was found to be highly model-dependent.

        vq_d: Subvector dimension for vector quantization.

        vq_bits: Bits per scalar weight parameter.
            Codebook size K = 2^(vq_bits * vq_d).
            WARNING: A common bug is K = 2^vq_bits, which gives only
            0.5 bits/param at vq_bits=2 (K=4 instead of K=256).

        vq_kmeans_niter: K-means iterations for codebook training.

        vptq_block_size: Block size for error propagation.
            Balances numerical stability (smaller blocks) vs speed (larger blocks).
            128 is standard.

        vptq_damp_percent: Hessian damping as fraction of mean diagonal.
            Prevents singular Hessian when some input features have
            near-zero variance. Added as damp * I before inversion.

        seed: Random seed for k-means initialization and data sampling.
    """

    k_factor: float = 1 / 8
    vq_d: int = 4
    vq_bits: int = 2
    vq_kmeans_niter: int = 20
    vptq_block_size: int = 128
    vptq_damp_percent: float = 0.01
    seed: int = 42

    @property
    def codebook_size(self) -> int:
        """K = 2^(vq_bits * vq_d). Number of centroids in the main codebook.

        For 2-bit quantization with d=4 subvectors: K = 2^(2*4) = 256.
        Each subvector index is 8 bits, encoding 2 bits per scalar weight.
        """
        return 2 ** (self.vq_bits * self.vq_d)
