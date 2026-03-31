import numpy as np
from option_pricer.util import Util

class BinomialModel():

    def price_american_option(n: int, 
                            S0: float,
                            K: float,
                            T: float, 
                            r: float, 
                            sigma: float, 
                            option_type: str
                            ) -> float:
        
        """
        Price an American option using a Cox-Ross-Rubinstein binomial tree.

        Parameters:
        n: Number of time steps in the tree.
        S0: Spot price.
        K: Strike price.
        T: Time to maturity in years.
        r: Continuously compounded risk-free rate.
        sigma: Volatility.
        option_type: "call" or "put".

        Returns: American option price.
        """
        if n <= 0:
            raise ValueError("n must be > 0")
        if S0 <= 0:
            raise ValueError("S0 must be > 0")
        if K <= 0:
            raise ValueError("K must be > 0")
        if T < 0:
            raise ValueError("T must be >= 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if T == 0:
            return float(Util.payoff(S0, K, option_type))

        dt = T/n
        up = np.exp(sigma* np.sqrt(dt))
        down = 1/up
        p = (np.exp(r * dt) - down)/(up-down)
        discount = np.exp(-r * dt)

        stock_prices = np.array([
            S0 * (up ** (n - i)) * (down ** i) for i in range(n + 1)
        ])

        option_values = Util.payoff(stock_prices, K, option_type)

        # backward induction
        for t in range(n - 1, -1, -1):
            stock_prices = np.array([
                S0 * (up ** (t - i)) * (down ** i) for i in range(t + 1)
            ])

            continuation_values = discount * (
                p * option_values[:t + 1] + (1 - p) * option_values[1:t + 2]
            )

            exercise_values = Util.payoff(stock_prices, K, option_type)
            option_values = np.maximum(continuation_values, exercise_values)

        return float(option_values[0])