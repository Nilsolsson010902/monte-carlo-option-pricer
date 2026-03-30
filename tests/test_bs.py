import math
import pytest
from mc_pricer.black_scholes import BlackScholes


def test_bs_call_known_value():
    price = BlackScholes.black_scholes_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_bs_put_call_parity():
    # Put-call parity: C - P = S0 - K*exp(-rT)
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    call = BlackScholes.black_scholes_price(S0, K, T, r, sigma, "call")
    put = BlackScholes.black_scholes_price(S0, K, T, r, sigma, "put")

    lhs = call - put
    rhs = S0 - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-6)


def test_bs_at_expiry_equals_intrinsic():
    call = BlackScholes.black_scholes_price(S0=95, K=100, T=0.0, r=0.05, sigma=0.2, option_type="call")
    put = BlackScholes.black_scholes_price(S0=95, K=100, T=0.0, r=0.05, sigma=0.2, option_type="put")

    assert call == 0.0
    assert put == 5.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        BlackScholes.black_scholes_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="banana")

def test_call_delta_between_zero_and_one():
    delta = BlackScholes.delta(S0=100,K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    assert 0.0 < delta < 1.0


def test_put_delta_between_zero_and_minus_one():
    delta = BlackScholes.delta(S0=100,K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    assert -1.0 < delta < 0.0


def test_call_delta_plus_put_delta_equals_one():
    call_delta = BlackScholes.delta(S0=100,K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    put_delta = BlackScholes.delta(S0=100,K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    assert abs((call_delta - put_delta) - 1.0) == 0

def test_vega_positive():
    vega = BlackScholes.vega(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
    assert vega > 0.0

def test_vega_zero_at_maturity():
    vega = BlackScholes.vega(S0=100, K=100, T=0.0, r=0.05, sigma=0.2)
    assert vega == 0.0

def test_delta_at_maturity_call():
    delta_itm = BlackScholes.delta(S0=120, K=100, T=0.0, r=0.05, sigma=0.2, option_type="call")
    delta_otm = BlackScholes.delta(S0=80, K=100, T=0.0, r=0.05, sigma=0.2, option_type="call")
    assert delta_itm == 1.0
    assert delta_otm == 0.0

def test_delta_at_maturity_put():
    delta_otm = BlackScholes.delta(S0=120, K=100, T=0.0, r=0.05, sigma=0.2, option_type="put")
    delta_itm = BlackScholes.delta(S0=80, K=100, T=0.0, r=0.05, sigma=0.2, option_type="put")
    assert delta_itm == -1.0
    assert delta_otm == 0.0

def test_gamma_positive():
    gamma = BlackScholes.gamma(S0=100, K=100, T=1.0, r=0.05,sigma=0.2)
    assert gamma > 0.0