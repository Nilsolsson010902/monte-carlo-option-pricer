from option_pricer.black_scholes import BlackScholes
from option_pricer.bopm import BinomialModel

def test_american_call_close_to_european_call_no_dividend():
    S0 = 100
    K = 100
    T = 1.0 
    r = 0.03
    sigma = 0.2
    am = BinomialModel.price_american_option(1000, S0, K, T, r, sigma, "call")
    eu = BlackScholes.black_scholes_price(S0, K, T, r, sigma, "call")
    assert abs(am - eu) < 0.2

def test_american_put_ge_european_put():
    S0 =100
    K = 100
    T = 1.0
    r = 0.03
    sigma = 0.2
    am = BinomialModel.price_american_option(1000, S0, K, T, r, sigma, "put")
    eu = BlackScholes.black_scholes_price(S0, K, T, r, sigma, "put")
    assert am >= eu

def test_bopm_converges():
    S0 = 100
    K = 100
    T = 1.0
    r = 0.03 
    sigma = 0.2 
    p1 = BinomialModel.price_american_option(100, S0, K, T, r, sigma, "put")
    p2 = BinomialModel.price_american_option(500, S0, K, T, r, sigma, "put")
    p3 = BinomialModel.price_american_option(1000, S0, K, T, r, sigma, "put")
    assert abs(p3 - p2) < abs(p2 - p1)