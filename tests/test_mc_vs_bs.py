import math
import pytest
from mc_pricer.black_scholes import BlackScholes
from mc_pricer.mc import MonteCarlo
import numpy as np


def test_bs_vs_mc_million_call():
    price_bs = BlackScholes.black_scholes_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    mc_call = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="call")
    _,ci = mc_call
    assert ci[0] < price_bs < ci[1]


def test_bs_vs_mc_million_put():
    price_bs = BlackScholes.black_scholes_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    mc_put= MonteCarlo.mc_simulator(n = 1000000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="put")
    _,ci = mc_put
    assert ci[0] < price_bs < ci[1]

def test_discount_logic_for_r():
    call_r_high = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="call")
    call_r_low = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=100, T=1.0, rf=0.01, sigma=0.2, option_type="call")

    put_r_high = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="put")
    put_r_low = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=100, T=1.0, rf=0.01, sigma=0.2, option_type="put")

    assert call_r_high> call_r_low and put_r_low > put_r_high

def test_t_zero():
    call = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=110, T=0, rf=0.05, sigma=0.2, option_type="call")
    put = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=110, T=0, rf=0.05, sigma=0.2, option_type="put")
    
    call_price,_ = call
    put_price,_ = put
    assert np.maximum(100 - 110, 0) == call_price and np.maximum(110-100,0) == put_price

def test_ci_converging():
    high_n = MonteCarlo.mc_simulator(n = 1000000, S0=100, K=110, T=1, rf=0.05, sigma=0.2, option_type="call")
    low_n = MonteCarlo.mc_simulator(n = 10000, S0=100, K=110, T=1, rf=0.05, sigma=0.2, option_type="call")

    _,high_ci = high_n
    _,low_ci = low_n

    assert (high_ci[1] - high_ci[0]) < (low_ci[1] - low_ci[0])