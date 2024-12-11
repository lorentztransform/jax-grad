import jax
import jax.numpy as jnp

class Value:
    """
    A class representing a value in our computational graph.
    This will be the basic building block for our automatic differentiation.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

def grad(f):
    """
    Returns a function that computes the gradient of f with respect to its input.
    
    Args:
        f: Function to differentiate
        
    Returns:
        Function that computes the gradient of f
    """
    def wrapped(x):
        x = Value(x)
        out = f(x)
        out.grad = 1.0
        out._backward()
        return x.grad
    return wrapped
