import numpy as np
from mc_pricer.black_scholes import BlackScholes
from mc_pricer.mc import MonteCarlo


def test_bs_vs_mc_call_ci_contains_bs():
    price_bs = BlackScholes.black_scholes_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    mc_price, ci = MonteCarlo.price_eu_option(n=200000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="call", method="plain",seed=42)
    assert ci[0] < price_bs < ci[1]


def test_bs_vs_mc_put_ci_contains_bs():
    price_bs = BlackScholes.black_scholes_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
    mc_price, ci = MonteCarlo.price_eu_option(n=200000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="put", method="plain", seed=42)
    assert ci[0] < price_bs < ci[1]


def test_interest_rate_logic():
    call_r_high, _ = MonteCarlo.price_eu_option(n=200000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="call", method="plain", seed=42)
    call_r_low, _ = MonteCarlo.price_eu_option(n=200000, S0=100, K=100, T=1.0, rf=0.01, sigma=0.2, option_type="call", method="plain", seed=42)
    put_r_high, _ = MonteCarlo.price_eu_option(n=200000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="put", method="plain",seed=42)
    put_r_low, _ = MonteCarlo.price_eu_option( n=200000, S0=100, K=100, T=1.0, rf=0.01,sigma=0.2, option_type="put", method="plain",seed=42)
    assert call_r_high > call_r_low
    assert put_r_low > put_r_high


def test_t_zero_matches_intrinsic_value():
    call_price, _ = MonteCarlo.price_eu_option( n=10000, S0=100, K=110, T=0.0, rf=0.05, sigma=0.2, option_type="call", method="plain",seed=42)
    put_price, _ = MonteCarlo.price_eu_option(n=10000,S0=100, K=110, T=0.0, rf=0.05, sigma=0.2,option_type="put", method="plain",seed=42)
    assert call_price == max(100 - 110, 0)
    assert put_price == max(110 - 100, 0)


def test_ci_width_shrinks_with_more_simulations():
    high_n_price, high_n_ci = MonteCarlo.price_eu_option(n=200000, S0=100, K=110, T=1.0, rf=0.05, sigma=0.2, option_type="call", method="plain", seed=42)
    low_n_price, low_n_ci = MonteCarlo.price_eu_option(n=10000, S0=100, K=110, T=1.0, rf=0.05, sigma=0.2, option_type="call", method="plain", seed=42)
    high_width = high_n_ci[1] - high_n_ci[0]
    low_width = low_n_ci[1] - low_n_ci[0]
    assert high_width < low_width


def test_antithetic_runs_and_returns_valid_ci():
    price, ci = MonteCarlo.price_eu_option(n=200000, S0=100, K=100, T=1.0, rf=0.05, sigma=0.2, option_type="call", method="antithetic",seed=42)
    assert price > 0
    assert ci[0] < ci[1]