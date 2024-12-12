"""Basic usage examples of JAX-Grad."""

from jax_grad.core.autodiff import Value, grad, forward_grad, DualNumber
from jax_grad.ops.math import sin, exp, log

def example_scalar_operations():
    """Example of basic scalar operations."""
    x = Value(2.0)
    y = Value(3.0)
    
    # Basic operations
    z = x * y + x
    print(f"Result: {z.data}")  # Should print 8.0
    
    # Gradient computation (reverse mode)
    z.backward()
    print(f"dx: {x.grad}")  # Should print 4.0 (dy/dx = y + 1)
    print(f"dy: {y.grad}")  # Should print 2.0 (dy/dy = x)

def example_forward_mode():
    """Example of forward-mode automatic differentiation."""
    def f(x):
        if isinstance(x, DualNumber):
            return x * x * x  # x³
        return x * x * x  # Handle regular numbers too
    
    x = 2.0
    # Compute derivative using forward mode
    dfdx = forward_grad(f)
    result = dfdx(x)
    print(f"d/dx(x³) at x=2: {result}")  # Should print 12.0 (3x²)

def example_math_operations():
    """Example of mathematical operations."""
    x = Value(1.0)
    
    # Compute y = sin(exp(x))
    y = sin(exp(x))
    y.backward()
    print(f"d/dx(sin(exp(x))) at x=1: {x.grad}")  # Should print cos(exp(1)) * exp(1)

if __name__ == "__main__":
    print("Basic scalar operations (reverse mode):")
    example_scalar_operations()
    print("\nForward mode differentiation:")
    example_forward_mode()
    print("\nMathematical operations:")
    example_math_operations()
