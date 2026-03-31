import numpy as np
import math
import scipy.stats as st

class Util():

    @staticmethod
    def standard_error(values: np.ndarray) -> float:
        """
        Standard error of the sample mean.
        """
        values = np.asarray(values)
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    
    @staticmethod
    def ci_normal(mean: float, std: float, level=0.95) -> tuple:
        """
        Calculates a confidence intervall for a given level
        mean: mean value
        sde: standard deviation
        level: CI level with default 95 %
        """
        alpha = 1.0 - level
        z_score = st.norm.ppf(1 - alpha / 2)
        upper = mean + z_score*std
        lower = mean - z_score*std
        return (lower, upper)
    
    @staticmethod
    def payoff(ST,  K: float, option_type: str):
        """
        payoff for call/put.
        """
        option_type = option_type.lower()

        if option_type == "call":
            return np.maximum(ST - K, 0)
        
        elif option_type == "put":
            return np.maximum(K - ST, 0)
        
        else:
            raise Exception("Invalid option type")
