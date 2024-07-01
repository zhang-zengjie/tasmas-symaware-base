import numpy as np
import os
from symawarebase.src.symaware.base import KnowledgeDatabase

root_path = os.getcwd()

class TasMasKnowledgeDatabase(KnowledgeDatabase):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    mu: np.ndarray
    Sigma: np.ndarray

    Q: np.ndarray
    R: np.ndarray
