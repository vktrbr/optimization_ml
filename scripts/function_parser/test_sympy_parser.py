import unittest


class MyTestCase(unittest.TestCase):
    def test_logarithm_replace_1(self):
        from sympy_parser import logarithm_replace
        test_input = 'log2(x + 1) + 2 * x'
        test_output = 'log(x + 1, 2) + 2 * x'
        self.assertEqual(logarithm_replace(test_input), test_output)

    def test_logarithm_replace_2(self):
        from sympy_parser import logarithm_replace
        test_input = '3*log2(x + 1)*log19(2*x) + log(x)'
        test_output = '3*log(x + 1, 2)*log(2*x, 19) + log(x)'
        self.assertEqual(logarithm_replace(test_input), test_output)

    def test_logarithm_replace_3(self):
        from sympy_parser import logarithm_replace
        test_input = 'log2(log215(loga(x))) + 50'
        test_output = 'log(log(log(x, a), 215), 2) + 50'
        self.assertEqual(logarithm_replace(test_input), test_output)


if __name__ == '__main__':
    unittest.main()
