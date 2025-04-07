### Live and Safe Marked Graph simulator class
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Any
from itertools import product
from MRF import MarkovRandomField

class LiveAndSafe:
    """
    Directed graph with tokens on edges representing activation points
    For our purposes it will be strictly strongly connected graphs

    Initialize with the following information:
    - V: vertices labeled 0 -> n
    - E: edges in an adjacency list
    - M_in: E -> {0, 1}
    """
    def __init__(self):
        self._vertices = set()   # V
        self._edges = {}         # E (as adjacency list)
        self._tokens = {}        # M dict((u, v) : {0, 1}) where 1 represents a token at edge (u, v)
        