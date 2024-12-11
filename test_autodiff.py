import unittest
from autodiff import Value, grad

class TestAutoDiff(unittest.TestCase):
    def test_simple_add(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        self.assertEqual(z.data, 5.0)
    
    def test_simple_mul(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        self.assertEqual(z.data, 6.0)
    
    def test_grad_square(self):
        def square(x):
            return x * x
        
        df = grad(square)
        result = df(3.0)  # derivative of x^2 is 2x, so at x=3 should be 6
        self.assertAlmostEqual(result, 6.0)

if __name__ == '__main__':
    unittest.main()
