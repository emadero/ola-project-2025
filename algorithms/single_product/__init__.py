
"""
Single Product Pricing Algorithms
Assigned to: Federico (Person 1)

This package contains algorithms for pricing a single product type.
"""

# Import algorithms when available
try:
    from .ucb import UCB1PricingAlgorithm
    from .constrained_ucb import UCBConstrainedPricingAlgorithm
    __all__ = ['UCB1PricingAlgorithm', 'UCBConstrainedPricingAlgorithm']
except ImportError:
    __all__ = []

# Will be added later:
# from .primal_dual import PrimalDualSingleProduct