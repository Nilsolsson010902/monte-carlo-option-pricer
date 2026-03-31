import math as math
class BlackScholes:

    
    @staticmethod
    def norm_cdf(x:float ) -> float:
        """
        The Normal Cumulative Distribution Function (CDF) calculates the probability that a normally 
        distributed random variable is less than or equal to a specific value x. 
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    @staticmethod
    def norm_pdf(x: float) -> float:
        """
        The normal probability density function (PDF) is a formula defining the bell-shaped curve of 
        continuous data centered around a mean. 
        """
        return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)
    
    @staticmethod
    def d1(S0: float, K: float, T: float, r: float, sigma: float) -> float:
        return (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(S0: float, K: float, T: float, r: float, sigma: float) -> float:
        return BlackScholes.d1(S0, K, T, r, sigma) - sigma * math.sqrt(T)

    @staticmethod
    def black_scholes_price(S0: float, 
                            K: float, 
                            T: float, 
                            r: float, 
                            sigma: float, 
                            option_type: str) -> float:
        """
            Black–Scholes price for a European option.
            Parameters:
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
            
        d1 = BlackScholes.d1(S0, K, T, r, sigma)
        d2 = BlackScholes.d2(S0, K, T, r, sigma)

        if option_type.lower() == "call":
            return S0*BlackScholes.norm_cdf(d1) - K*BlackScholes.norm_cdf(d2)*math.exp(-r * T)
        
        elif option_type.lower() == "put":
            return K * math.exp(-r * T) * BlackScholes.norm_cdf(-d2) - S0 * BlackScholes.norm_cdf(-d1)

        else:
            raise ValueError("option_type must be 'call' or 'put'")
        

    @staticmethod
    def delta(S0: float,
                K: float,
                T: float,
                r: float,
                sigma: float,
                option_type: str
                )-> float:
        """ 
        Delta greek for european call option.
        Measures relationship between options price and underlying stock. 
        """

        if T <= 0:
            if option_type.lower() == "call":
                return 1.0 if S0 > K else 0.0
            elif option_type.lower() == "put":
                return -1.0 if S0 < K else 0.0          #put option delta is -1.0 when ITM
            else:
                raise ValueError("No such option type")
            
        d1 = BlackScholes.d1(S0, K, T, r, sigma)

        if option_type.lower() == "call":
            return BlackScholes.norm_cdf(d1)
        elif option_type.lower() == "put":
            return BlackScholes.norm_cdf(d1) - 1.0
        else:
            raise ValueError("No such option type")

    @staticmethod 
    def vega(S0: float,
              K: float,
              T: float,
              r: float,
              sigma: float,
              )-> float:
        """ 
        Vega greek for european call option.
        Measures the option's price sensitivity to changes in the underlying asset's implied volatility
        """
        if T <= 0:
            return 0.0

        d1 = BlackScholes.d1(S0, K, T, r, sigma)
        return S0 * BlackScholes.norm_pdf(d1) * math.sqrt(T)


    @staticmethod
    def gamma(S0: float,
              K: float,
              T: float,
              r: float,
              sigma: float,
              )-> float:
        """ 
        Gamma greek for european call option.
        Gamma measures the sensitivity of the option's delta.
        """
        if T <= 0:
            return 0.0

        d1 = BlackScholes.d1(S0, K, T, r, sigma)
        return BlackScholes.norm_pdf(d1) / (S0 * sigma * math.sqrt(T))


if __name__ == "__main__":
    print(BlackScholes.black_scholes_price(100, 95, 0.25, 0.1 , 0.5, "call"))
