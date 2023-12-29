from scipy.stats import norm
from numpy import exp, log,sqrt

N = norm.cdf
n = norm.pdf

def is_numeric(value):
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


def GetPremium(S,T,r,K,sigma,flag='c'):
    '''Find the Black Scholes price of a European Call Option
        S: Stock Price at time 0
        T: Time to expiration (annualized)
        r: Annual risk free rate
        K: Strike Price
        sigma: Annualized Volatility
        flag: option type, 'c' or 'p'
    '''
    for i, val in enumerate([S,T,r,K,sigma]):
        if is_numeric(val) is False: raise TypeError(f"Inputs should be numeric: {val}")
        if val <= 0: raise ValueError(f"Inputs should be greater than zero: {['S','T','r','K','sigma'][i]} = {val}")

    d1 = (log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if flag == 'c':
        return S*N(d1) - K*exp(-r*T)*N(d2)
    else:
        return -S*N(-d1) + K*exp(-r*T)*N(-d2)


def GetVega(S,T,r,K,sigma, flag='c'):
    '''Find the "Vega" of a Black Scholes European Call Option
    S: Stock Price at time 0
    T: Time to expiration (annualized)
    r: Annual risk free rate
    K: Strike Price
    sigma: Annualized Volatility
    flag: option type, 'c' or 'p'
    '''
    for i, val in enumerate([S,T,r,K,sigma]):
        if is_numeric(val) is False: raise TypeError(f"Inputs should be numeric: {val}")
        if val <= 0: raise ValueError(f"Inputs should be greater than zero: {['S','T','r','K','sigma'][i]} = {val}")
    
    d1 = (log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    if flag == 'c':
        return S*sqrt(T)*n(d1)
    else:
        return S*sqrt(T)*n(-d1)


def GetIV(target_value,S,T,r,K,sigma_guess=.5,flag='c'):
    '''Estimate the Implied Volatility of a European Option using Newton's Method.
    target_value: The price of the option
    S: Stock Price at time 0
    T: Time to expiration (annualized)
    r: Annual risk free rate
    K: Strike Price
    sigma: Annualized Volatility
    flag: option type, 'c' or 'p'
    Output: {'IV': sigma, 'Numer of Iterations': max_iterations}
    '''
    for i, val in enumerate([target_value,S,T,r,K,sigma_guess]):
        if is_numeric(val) is False: raise TypeError(f"Inputs should be numeric: {val}")
        if val <= 0: raise ValueError(f"Inputs should be greater than zero: {['S','T','r','K','sigma'][i]} = {val}")


    sigma = sigma_guess
    max_iterations = 200
    precision = 1e-5

    for i in range(max_iterations):
        price = GetPremium(S,T,r,K,sigma,flag)
        vega = GetVega(S,T,r,K,sigma)

        diff = target_value - price

        if abs(diff) < precision:
            return {'IV': sigma, 'Number of Iterations': i}
        
        #Use Newton Method to find next step. Formula is x_n = x_{n-1} - f(sigma)/f'(sigma). Here, f(sigma) = diff
        #Plus sign comes from the definition of diff. Derivitive of diff WRT sigma causes double negative
        #Times 10 factor helps with stable convergence
        sigma = abs(sigma + diff/(10*vega))
        
        if sigma < 1e-4:
            sigma = sigma_guess/2

    return {'IV': sigma, 'Numer of Iterations': max_iterations}