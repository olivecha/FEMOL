import unittest
import sys
import numpy as np
from scipy import integrate
sys.path.append('../')
from FEMOL.elements import  Q8

class MyTestCase(unittest.TestCase):
    """
    Test class for the Q8 Element class
    """
    def test_3rd_order_gauss_quadrature(self):
        """
        Tests the 3rd order gauss quadrature of the Q8 element
        """
        def fun(y, x):
            """
            Integration test function
            """
            return x**3 + y**3

        element_integral = 0
        for pt, w in zip(Q8.integration_points_3, Q8.integration_weights_3):
            element_integral += w * fun(*pt)

        reference_integral = integrate.dblquad(fun, -1, 1, lambda x: -1, lambda x: 1)

        self.assertTrue(np.isclose(element_integral, reference_integral[0]))

if __name__ == '__main__':
    unittest.main()
