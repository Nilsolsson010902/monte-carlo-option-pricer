import math
import numpy as np
from mc_pricer.stats import Stats

class MonteCarlo:

    @staticmethod
    def gbm_simulation(S0: float, rf: float, sigma: float, T: float) -> float:
        """
            Geometric Brownian Motion formula for stock price. 
            Inputs:
            S0: spot price today
            rf: continuously compounded risk-free rate (e.g. 0.02)
            sigma: volatility (e.g. 0.2)
            T: future date for stock price.
        """
        z = MonteCarlo.generate_z()
        return  S0 * np.exp((rf - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)



    @staticmethod
    def generate_z() -> float:
        #Generates a random number z from a norm distribution N(0,1)
        return np.random.normal(0, 1)
    

    @staticmethod
    def mc_discount(price: float, rf: float, T: float) -> float:
        """
        Discount model for option price
        Price: future price
        rf: continuously compounded risk-free rate (e.g. 0.02)
        T: time corresponding with future price.
        """
        return price*np.exp(-rf*T)
    
    @staticmethod
    def payoff(ST: float, K: float, option_type: str) -> float:
        if option_type == "call":
            return np.maximum(ST - K, 0)
        
        elif option_type == "put":
            return np.maximum(K - ST, 0)
        
        else:
            raise Exception("Invalid option type")

    @staticmethod
    def ST_simulator(n: int, S0: float, T: float, rf: float, sigma: float) -> list:
        st = []
        for i in range(0, n):
            gbm_price = MonteCarlo.gbm_simulation(S0, rf,sigma, T)
            st.append(gbm_price)
        return st

    @staticmethod
    def mc_simulator(n: int, S0: float, K: float, T: float, rf: float, sigma: float, option_type: str) -> tuple:
        """
        Monte Carlo simulation for calculating price of call and put option.
        n: number of simulations
        S0: spot price today
        K: strike price
        T: future date for stock price.
        rf: continuously compounded risk-free rate (e.g. 0.02)
        sigma: volatility (e.g. 0.2)
        option_type: call or put
        """
        stock_prices = MonteCarlo.ST_simulator(n, S0, T, rf, sigma)
        payoffs = [MonteCarlo.payoff(price, K, option_type) for price in stock_prices]
        mean_payoff = np.mean(payoffs)
        price = MonteCarlo.mc_discount(mean_payoff, rf, T)
        std = MonteCarlo.mc_discount(Stats.std_from_sum(payoffs), rf, T)
        ci = Stats.ci_normal(price, std)
        return price, ci
