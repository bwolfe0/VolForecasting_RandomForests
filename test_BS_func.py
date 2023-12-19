import unittest
import BS_functions

class  Test_BS_func(unittest.TestCase):

    #Test source is quantpy.com
    def test_bs_price(self):
        result = BS_functions.GetCall(S=30,K=28,T=.2,r=.025,sigma=.53816)
        self.assertAlmostEqual(result,3.9700874184482906, places=4)

    def test_GetVega(self):
        result = BS_functions.GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816)
        self.assertAlmostEqual(0.04884,result*.01,places=4)

    def test_find_vol(self):
        result = BS_functions.GetIV(target_value=3.97,S=30,K=28,T=.2,r=.025,sigma_guess=.5)
        self.assertAlmostEqual(.53816,result['Sigma'],places=4)

if __name__ == "__main__":
    unittest.main()