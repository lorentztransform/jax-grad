import jax
import jax.numpy as jnp

class DualNumber:
    """
    A class representing a dual number for forward-mode automatic differentiation.
    A dual number has the form a + bε where ε² = 0.
    The real part 'a' represents the value, and the dual part 'b' represents the derivative.
    """
    def __init__(self, real, dual=0.0):
        self.real = float(real)
        self.dual = float(dual)
    
    def __repr__(self):
        return f"DualNumber(real={self.real}, dual={self.dual})"
    
    def __add__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(float(other))
        return DualNumber(self.real + other.real, self.dual + other.dual)
    
    def __mul__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(float(other))
        return DualNumber(self.real * other.real, 
                         self.real * other.dual + self.dual * other.real)
    
    def __truediv__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(float(other))
        return DualNumber(self.real / other.real,
                         (self.dual * other.real - self.real * other.dual) / (other.real ** 2))
    
    def __sub__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(float(other))
        return DualNumber(self.real - other.real, self.dual - other.dual)
    
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return DualNumber(other) - self
    def __rtruediv__(self, other): return DualNumber(other) / self
    def __neg__(self): return DualNumber(-self.real, -self.dual)

def forward_grad(f):
    """
    Returns a function that computes the derivative of f using forward-mode AD.
    
    Args:
        f: Function to differentiate
        
    Returns:
        Function that computes the derivative of f using forward-mode AD
    """
    def wrapped(x):
        # Create dual number with dual part 1.0 to compute derivative
        dual_x = DualNumber(float(x), 1.0)
        result = f(dual_x)
        return result.dual if isinstance(result, DualNumber) else 0.0
    return wrapped

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

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')
        
        def _backward():
            # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        
        return out

    def __rtruediv__(self, other):
        return Value(other) / self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power only supports int/float exponents for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out

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
