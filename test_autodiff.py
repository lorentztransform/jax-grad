import unittest
from autodiff import Value, grad

class TestAutoDiff(unittest.TestCase):
    def test_add(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        self.assertEqual(z.data, 5.0)
        z.backward()
        self.assertEqual(x.grad, 1.0)  # dz/dx = 1
        self.assertEqual(y.grad, 1.0)  # dz/dy = 1
    
    def test_mul(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        self.assertEqual(z.data, 6.0)
        z.backward()
        self.assertEqual(x.grad, 3.0)  # dz/dx = y = 3
        self.assertEqual(y.grad, 2.0)  # dz/dy = x = 2
    
    def test_sub(self):
        x = Value(5.0)
        y = Value(2.0)
        z = x - y
        self.assertEqual(z.data, 3.0)
        z.backward()
        self.assertEqual(x.grad, 1.0)   # dz/dx = 1
        self.assertEqual(y.grad, -1.0)  # dz/dy = -1
    
    def test_mixed_operations(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + y
        self.assertEqual(z.data, 9.0)  # (2 * 3) + 3 = 9
        z.backward()
        self.assertEqual(x.grad, 3.0)   # dz/dx = y = 3
        self.assertEqual(y.grad, 3.0)   # dz/dy = x + 1 = 3
    
    def test_grad_square(self):
        def square(x):
            return x * x
        
        df = grad(square)
        result = df(3.0)  # derivative of x^2 is 2x, so at x=3 should be 6
        self.assertAlmostEqual(result, 6.0)
    
    def test_grad_quadratic(self):
        def quadratic(x):
            return x * x + 2 * x + 1  # x^2 + 2x + 1
        
        df = grad(quadratic)
        result = df(2.0)  # derivative is 2x + 2, at x=2 should be 6
        self.assertAlmostEqual(result, 6.0)

if __name__ == '__main__':
    unittest.main()
