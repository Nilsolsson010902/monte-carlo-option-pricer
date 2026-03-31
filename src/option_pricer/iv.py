import numpy as np
from option_pricer.black_scholes import BlackScholes
class ImpliedVol:

    @staticmethod
    def newton_method(market_price: float,
                    S0: float,
                    K: float,
                    T: float,
                    r: float,
                    option_type: str,
                    initial_guess: float = 0.2,
                    goal: float = 1e-8,
                    max_iter: int = 100,
                    )  -> float:
        """
        Calculates the implied volatility by optimizing the initial guess with black scholes price and vega.
        Parameters:
        market_price: the market price of the option.
        S0: spot price. 
        K: strike price.
        T: time to maturity in years. 
        r: continuously compounded risk-free rate. 
        option_type: "call" or "put".
        initial_guess: start guess for volatility - it has to start somwhere in newton.
        goal: goal boundary
        max_iter: max iteration to prevent infinity loop
        """
        sigma = initial_guess

        for i in range(max_iter):
            bs_price = BlackScholes.black_scholes_price(S0, K, T, r, sigma, option_type)
            vega = BlackScholes.vega(S0, K, T, r, sigma)

            price_diff = bs_price - market_price
            if(abs(price_diff) < goal):
                return sigma

            # If Vega is too small, Newton step becomes unstable
            if vega < 1e-12:
                raise ValueError("Vega too small for Newton method")
            
            sigma = sigma - (price_diff)/vega

            if sigma <= 0: #Keep sigma positive to avoid invalid volatility values
                sigma = 1e-6
            
        return sigma