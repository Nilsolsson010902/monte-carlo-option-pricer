import math as math
from datetime import date
import numpy as np
from mc_pricer.mc import MonteCarlo
class BlackScholes:

    
    @staticmethod
    def norm_cdf(x:float ) -> float:
        # Standard normal CDF via error function
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def black_scholes_price(S0: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """
            Black–Scholes price for a European option.
            Inputs:
            S0: spot price today
            K: strike
            T: time to maturity in years (e.g. 0.5 = 6 months)
            r: continuously compounded risk-free rate (e.g. 0.02)
            sigma: volatility (e.g. 0.2)
            option_type: "call" or "put"
        """
        if T <= 0:
            if option_type.lower() == "call":
                return max(S0 - K, 0.0)
            elif option_type.lower() == "put":
                return max(K-S0, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
    
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
            
        d1 = (math.log(S0/K) + (r + sigma**2 * 0.5 ) * T)/(sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)

        if option_type.lower() == "call":
            return S0*BlackScholes.norm_cdf(d1) - K*BlackScholes.norm_cdf(d2)*math.exp(-r * T)
        
        elif option_type.lower() == "put":
            return K * math.exp(-r * T) * BlackScholes.norm_cdf(-d2) - S0 * BlackScholes.norm_cdf(-d1)

        else:
            raise ValueError("option_type must be 'call' or 'put'")

if __name__ == "__main__":
    print(BlackScholes.black_scholes_price(100, 95, 0.25, 0.1 , 0.5, "call"))
    print(MonteCarlo.mc_simulator(100000, 100, 95, 0.25, 0.1 , 0.5, "put"))