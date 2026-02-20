import math
import numpy as np

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
    def call_option_max(ST: float, K: float) -> float:
        """
        Calculates call option profit.
        ST: Current stock price
        K: Strike price
        """
        return np.maximum(ST - K, 0)
    
    @staticmethod
    def put_option_max(ST: float, K: float) -> float:
        """
        Calculates put option profit.
        ST: Current stock price
        K: Strike price
        """
        return np.maximum(K - ST, 0)

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
    def mc_simulator(n: int, S0: float, K: float, T: float, rf: float, sigma: float) -> tuple:
        """
        Monte Carlo simulation for calculating price of call and put option.
        n: number of simulations
        S0: spot price today
        K: strike price
        T: future date for stock price.
        rf: continuously compounded risk-free rate (e.g. 0.02)
        sigma: volatility (e.g. 0.2)
        """
        call_options = 0
        put_options = 0
        for i in range(0, n):
            gbm_price = MonteCarlo.gbm_simulation(S0, rf,sigma, T)
            c_price = MonteCarlo.call_option_max(gbm_price, K)
            p_price = MonteCarlo.put_option_max(gbm_price, K)
            call_options += c_price
            put_options += p_price
        
        average_c_price = call_options/n
        average_p_price = put_options/n

        discounted_c = MonteCarlo.mc_discount(average_c_price, rf,  T)
        discounted_p = MonteCarlo.mc_discount(average_p_price, rf,  T)


        return(discounted_c, discounted_p)
