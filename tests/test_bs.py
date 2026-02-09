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
