import unittest
import BS_functions
from numpy import exp

class  Test_BS_func(unittest.TestCase):

    #Test source is quantpy.com
    def test_bs_price(self):
        self.assertAlmostEqual(3.97009, BS_functions.GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816), places=4)
        with self.assertRaises(ValueError):
            BS_functions.GetPremium(S=-5,K=28,T=.2,r=.025,sigma=.53816)

    def test_put_call_parity(self):
        self.assertAlmostEqual(BS_functions.GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='c') - BS_functions.GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='p'),
                               30 - 28*exp(-.2*.025), places=4)

    def test_GetVega(self):
        self.assertAlmostEqual(0.04884,BS_functions.GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816)*.01,places=4)
        with self.assertRaises(ValueError):
            BS_functions.GetVega(S=-5,K=28,T=.2,r=.025,sigma=.53816)

    def test_find_vol(self):
        self.assertAlmostEqual(.53816,BS_functions.GetIV(target_value=3.97,S=30,K=28,T=.2,r=.025,sigma_guess=.5)['IV'],places=4)
        with self.assertRaises(ValueError):
            BS_functions.GetIV(target_value=3.97,S=-5,K=28,T=.2,r=.025,sigma_guess=.53816)


if __name__ == "__main__":
    unittest.main()