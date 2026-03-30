from mc_pricer.black_scholes import BlackScholes
from mc_pricer.iv import ImpliedVol


def test_implied_vol_recovers_true_sigma_call():
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    true_sigma = 0.2

    market_price = BlackScholes.black_scholes_price(S0=S0, K=K, T=T, r=r, sigma=true_sigma, option_type="call")
    implied_vol = ImpliedVol.newton_method(market_price=market_price, S0=S0, K=K, T=T, r=r, option_type="call")
    assert abs(implied_vol - true_sigma) < 1e-6


def test_implied_vol_recovers_true_sigma_put():
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    true_sigma = 0.2

    market_price = BlackScholes.black_scholes_price(S0=S0, K=K, T=T, r=r, sigma=true_sigma, option_type="put")
    implied_vol = ImpliedVol.newton_method(market_price=market_price, S0=S0, K=K, T=T, r=r, option_type="put")
    assert abs(implied_vol - true_sigma) < 1e-6


def test_implied_vol_with_non_default_initial_guess():
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    true_sigma = 0.25

    market_price = BlackScholes.black_scholes_price(S0=S0,K=K, T=T, r=r, sigma=true_sigma, option_type="call")
    implied_vol = ImpliedVol.newton_method(market_price=market_price, S0=S0, K=K, T=T, r=r, option_type="call",initial_guess=0.5)
    assert abs(implied_vol - true_sigma) < 1e-6


def test_repriced_option_matches_market_price_after_iv_inversion():
    S0 = 100
    K = 105
    T = 0.75
    r = 0.03
    true_sigma = 0.22

    market_price = BlackScholes.black_scholes_price(S0=S0, K=K, T=T, r=r, sigma=true_sigma, option_type="call")
    implied_vol = ImpliedVol.newton_method( market_price=market_price, S0=S0, K=K, T=T, r=r, option_type="call")
    repriced = BlackScholes.black_scholes_price(S0=S0, K=K, T=T, r=r, sigma=implied_vol, option_type="call")
    assert abs(repriced - market_price) < 1e-8
