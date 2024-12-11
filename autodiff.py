import jax
import jax.numpy as jnp

class Value:
    """
    A class representing a value in our computational graph.
    This will be the basic building block for our automatic differentiation.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)  # Convert to float for consistent operations
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        
        def _backward():
            # d(a-b)/da = 1, d(a-b)/db = -1
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward
        
        return out

    def __rsub__(self, other):
        return Value(other) - self

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        """
        Performs backward pass to compute gradients.
        """
        # Initialize grad of output to 1.0
        self.grad = 1.0
        
        # Call backward on this node
        self._backward()

def grad(f):
    """
    Returns a function that computes the gradient of f with respect to its input.
    
    Args:
        f: Function to differentiate
        
    Returns:
        Function that computes the gradient of f
    """
    def wrapped(x):
        if not isinstance(x, Value):
            x = Value(x)
        # Forward pass
        out = f(x)
        # Backward pass
        out.backward()
        return x.grad
    return wrapped
