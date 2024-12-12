"""
JAX-Grad: A from-scratch implementation of automatic differentiation using JAX.
"""

from jax_grad.core.autodiff import Value, grad

__version__ = '0.1.0'
__all__ = ['Value', 'grad']
