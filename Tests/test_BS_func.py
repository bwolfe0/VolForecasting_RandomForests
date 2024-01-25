# import sys
# sys.path.append(sys.path[0][:-6])

import unittest
from Functions.BS_functions import *
from numpy import exp


class  Test_BS_func(unittest.TestCase):

    #Test source is quantpy.com
    def test_GetPremium(self):
        self.assertAlmostEqual(3.97009, GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816), places=4) #Test known solution
        with self.assertRaises(ValueError):
            GetPremium(S=-5,K=28,T=.2,r=.025,sigma=.53816) #Test negative input

    def test_put_call_parity(self):
        self.assertAlmostEqual(GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='c') - 
                               GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='p'),
                               30 - 28*exp(-.2*.025), places=4) #Test PCP equation

    def test_GetVega(self):
        self.assertAlmostEqual(0.04884,GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816)*.01,places=4) #Test known solution
        self.assertAlmostEqual(GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='c'),GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='p'),places=4) #Test Vega_call=Vega_put
        with self.assertRaises(ValueError):
            GetVega(S=-5,K=28,T=.2,r=.025,sigma=.53816) #Test negative input

    def test_GetIV(self):
        self.assertAlmostEqual(.53816,GetIV(target_value=3.97,S=30,K=28,T=.2,r=.025,sigma_guess=.5)['IV'],places=4) #Test known solution
        with self.assertRaises(ValueError):
            GetIV(target_value=3.97,S=-5,K=28,T=.2,r=.025,sigma_guess=.53816) #Test negative input


if __name__ == "__main__":
    unittest.main()