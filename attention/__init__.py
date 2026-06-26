from .dense import dense_attention
from .dsa   import dsa_attention
from .csa   import csa_attention
from .hca   import hca_attention

REGISTRY = {
    "dense": dense_attention,
    "dsa":   dsa_attention,
    "csa":   csa_attention,
    "hca":   hca_attention,
}

__all__ = ["dense_attention", "dsa_attention", "csa_attention", "hca_attention", "REGISTRY"]
