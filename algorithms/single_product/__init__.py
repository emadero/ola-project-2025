# algorithms/single_product/__init__.py
"""
Single Product Pricing Algorithms

This package contains algorithms for pricing a single product type.
"""

# Import algorithms when available
try:
    from .ucb import UCB1Algorithm
    __all__ = ['UCB1Algorithm']
except ImportError:
    __all__ = []

# Will be added later:
# from .constrained_ucb import ConstrainedUCB1Algorithm  
# from .primal_dual import PrimalDualSingleProduct