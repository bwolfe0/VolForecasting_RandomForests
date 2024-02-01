import pytest
from Functions.BS_functions import *
from numpy import exp


def test_GetPremium():
        assert pytest.approx(3.97009, rel=1e-3) == GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816) #Test known solution
        with pytest.raises(ValueError):
            GetPremium(S=-5,K=28,T=.2,r=.025,sigma=.53816) #Test negative input

def test_put_call_parity():
        assert pytest.approx(GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='c') - 
                               GetPremium(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='p'), rel=1e-3) == 30 - 28*exp(-.2*.025) #Test PCP equation

def test_GetVega():
        assert pytest.approx(0.04884, rel=1e-3)  == GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816)*.01 #Test known solution
        assert pytest.approx(GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='c')) == GetVega(S=30,K=28,T=.2,r=.025,sigma=.53816,flag='p', rel=1e-3) #Test Vega_call=Vega_put
        with pytest.raises(ValueError):
            GetVega(S=-5,K=28,T=.2,r=.025,sigma=.53816) #Test negative input

def test_GetIV():
        assert pytest.approx(.53816, rel=1e-3) == GetIV(target_value=3.97,S=30,K=28,T=.2,r=.025,sigma_guess=.5)['IV'] #Test known solution
        with pytest.raises(ValueError):
            GetIV(target_value=3.97,S=-5,K=28,T=.2,r=.025,sigma_guess=.53816) #Test negative input