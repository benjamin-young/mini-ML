import unittest
from engine.shapes import Tensor

import numpy as np


class TestTensor(unittest.TestCase):
    def test_tensor_mul(self):
        t1 = Tensor(np.array([[1,2],[3,4]]))
        t2 = Tensor(np.array([[5,6],[7,8]]))
        expected_result = np.array([[19, 22], [43, 50]])
        t3 = t1 * t2
        print(t3.values)
        np.testing.assert_array_equal(t3.values, expected_result)

if __name__ == '__main__':
    unittest.main()