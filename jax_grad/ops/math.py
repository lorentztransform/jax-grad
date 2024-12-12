"""Mathematical operations for automatic differentiation."""

import numpy as np
from ..core.autodiff import Value

def exp(x):
    """Exponential function."""
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.exp(x.data), (x,), 'exp')
    
    def _backward():
        # d(e^x)/dx = e^x
        x.grad += np.exp(x.data) * out.grad
    out._backward = _backward
    
    return out

def log(x):
    """Natural logarithm."""
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.log(x.data), (x,), 'log')
    
    def _backward():
        # d(ln(x))/dx = 1/x
        x.grad += (1.0 / x.data) * out.grad
    out._backward = _backward
    
    return out

def sin(x):
    """Sine function."""
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.sin(x.data), (x,), 'sin')
    
    def _backward():
        # d(sin(x))/dx = cos(x)
        x.grad += np.cos(x.data) * out.grad
    out._backward = _backward
    
    return out

def cos(x):
    """Cosine function."""
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.cos(x.data), (x,), 'cos')
    
    def _backward():
        # d(cos(x))/dx = -sin(x)
        x.grad += -np.sin(x.data) * out.grad
    out._backward = _backward
    
    return out

def tan(x):
    """Tangent function."""
    x = x if isinstance(x, Value) else Value(x)
    out = Value(np.tan(x.data), (x,), 'tan')
    
    def _backward():
        # d(tan(x))/dx = sec^2(x) = 1 + tan^2(x)
        x.grad += (1.0 + np.tan(x.data)**2) * out.grad
    out._backward = _backward
    
    return out
