### Live and Safe Marked Graph simulator class
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Any
from itertools import product
from MRF import MarkovRandomField

# Base class
class LiveAndSafe:
    """
    Directed graph with tokens on edges representing activation points
    For our purposes it will be strictly strongly connected graphs

    Initialize with the following information:
    - 
    """
    def __init__(self):
