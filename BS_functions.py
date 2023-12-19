from scipy.stats import norm
from numpy import exp, log,sqrt

N = norm.cdf
n = norm.pdf


def GetCall(S,T,r,K,sigma):
    '''Find the Black Scholes price of a European Call Option
        S: Stock Price at time 0
        T: Time to expiration (annualized)
        r: Annual risk free rate
        K: Strike Price
        sigma: Annualized Volatility
    '''
    d1 = (log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S*N(d1) - K*exp(-r*T)*N(d2)

def GetVega(S,T,r,K,sigma):
    '''Find the "Vega" of a Black Scholes European Call Option
    S: Stock Price at time 0
    T: Time to expiration (annualized)
    r: Annual risk free rate
    K: Strike Price
    sigma: Annualized Volatility
    '''
    d1 = (log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    return S*sqrt(T)*n(d1) 

def GetIV(target_value,S,T,r,K,sigma_guess):
    '''Estimate the Implied Volatility of a European Option using Newton's Method.
    target_value: The price of the option
    S: Stock Price at time 0
    T: Time to expiration (annualized)
    r: Annual risk free rate
    K: Strike Price
    sigma: Annualized Volatility
    '''
    sigma = sigma_guess
    max_iterations = 200
    precision = 1e-5

    for i in range(max_iterations):
        price = GetCall(S,T,r,K,sigma)
        vega = GetVega(S,T,r,K,sigma)

        diff = target_value - price

        if abs(diff) < precision:
            return {'Sigma': sigma, 'Number of Iterations': i}
        
        #Use Newton Method to find next step. Formula is x_n = x_{n-1} - f(sigma)/f'(sigma). Here, f(sigma) = diff
        #Plus sign comes from the definition of diff. Derivitive of diff WRT sigma causes double negative
        sigma = sigma + diff/vega 

    return sigma