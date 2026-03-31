import numpy as np
from option_pricer.util import Util

class MonteCarlo():

    @staticmethod
    def terminal_price_simulation(n: int, 
                                    S0: float, 
                                    r: float, 
                                    sigma: float, 
                                    T: float,
                                    method: str = "plain",
                                    seed: int| None = None
                                    ) -> np.ndarray:
        """
            Simulates terminal stockpices with risk neutral GBM. 
            Parameters:
            n: Number of simulations.
            S0: Spot price.
            r: Continuously compounded risk-free rate.
            sigma: Volatility.
            T: Future date for stock price (maturity).
            method: "plain" or "antithetic".
            seed: Random seed for reproducibility.

            Returns: Array of simulated terminal prices.
        """
        if n <= 0:
            raise ValueError("n must be > 0")
        
        Util.check_option_parameters(S0, T, sigma)
        rand_num = np.random.default_rng(seed)

        if method == "plain":
            z = rand_num.standard_normal(n)

        elif method == "antithetic":
            half_n = (n + 1) // 2
            z_half = rand_num.standard_normal(half_n)
            z = np.concatenate([z_half, -z_half])[:n]

        else:
            raise ValueError("method must be 'plain' or 'antithetic'")
        
        return  S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


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
        n: Number of simulations
        S0: Spot price today
        K: Strike price
        T: Future date for stock price.
        rf: Continuously compounded risk-free rate (e.g. 0.02)
        sigma: Volatility (e.g. 0.2)
        option_type: Call or put
        method: "plain" or "antithetic" 
        seed: Random seed for reproducibility.

        Returns: Monte Carlo price for European option with confidence interval.
        """

        ST = MonteCarlo.terminal_price_simulation(n=n, S0=S0, T=T, r=rf, sigma=sigma, method=method, seed=seed,)
        payoffs = Util.payoff(ST, K, option_type)
        discounted_payoffs = np.exp(-rf * T) * payoffs
        price = float(np.mean(discounted_payoffs))
        stderr = Util.standard_error(discounted_payoffs)
        ci = Util.ci_normal(price, stderr, level=confidence_level)
        return price, ci
