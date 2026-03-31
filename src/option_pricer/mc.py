import numpy as np
from option_pricer.util import Util


class MonteCarlo():

    @staticmethod
    def terminal_price_simulation(n: int, 
                                  S0: float, 
                                  rf: float, 
                                  sigma: float, 
                                  T: float,
                                  method: str = "plain",
                                  seed: int| None = None
                                  ) -> np.ndarray:
        """
            Simulates terminal stockpices with risk neutral GBM. 
            Parameters:
            n: number of simulations.
            S0: spot price.
            rf: continuously compounded risk-free rate.
            sigma: volatility.
            T: future date for stock price (maturity).
            method:"plain" or "antithetic".
            seed: random seed for reproducibility.

            Returns: Array of simulated terminal prices.
        """
        if n <= 0:
            raise ValueError("n must be > 0")
        if S0 <= 0:
            raise ValueError("S0 must be > 0")
        if T < 0:
            raise ValueError("T must be >= 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        
        rand_num = np.random.default_rng(seed)

        if method == "plain":
            z = rand_num.standard_normal(n)

        elif method == "antithetic":
            half_n = (n + 1) // 2
            z_half = rand_num.standard_normal(half_n)
            z = np.concatenate([z_half, -z_half])[:n]

        else:
            raise ValueError("method must be 'plain' or 'antithetic'")
        
        return  S0 * np.exp((rf - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


    @staticmethod
    def price_eu_option(n: int, 
                        S0: float, 
                        K: float, 
                        T: float, 
                        rf: float, 
                        sigma: float, 
                        option_type: str,
                        method: str = "plain",
                        confidence_level: float = 0.95,
                        seed: int | None = None,
                        ) -> tuple[float, tuple[float, float]]:
        """
        Monte Carlo simulation for calculating price of call and put option.
        n: number of simulations
        S0: spot price today
        K: strike price
        T: future date for stock price.
        rf: continuously compounded risk-free rate (e.g. 0.02)
        sigma: volatility (e.g. 0.2)
        option_type: call or put
        method: "plain" or "antithetic" 
        seed: random seed for reproducibility.

        Returns: Monte Carlo price for European option with confidence interval.
        """

        ST = MonteCarlo.terminal_price_simulation(n=n, S0=S0, T=T, rf=rf, sigma=sigma, method=method, seed=seed,)
        payoffs = Util.payoff(ST, K, option_type)
        discounted_payoffs = np.exp(-rf * T) * payoffs
        price = float(np.mean(discounted_payoffs))
        stderr = Util.standard_error(discounted_payoffs)
        ci = Util.ci_normal(price, stderr, level=confidence_level)
        return price, ci
