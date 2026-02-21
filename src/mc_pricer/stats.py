import numpy as np
import math
import scipy.stats as st

class Stats():

    @staticmethod
    def std_from_sum(sumseq: list) -> float:
        #Calculates standard deviation from a list of numbers
        mean = np.mean(sumseq)
        std = np.std(sumseq, ddof=1)
        return std / np.sqrt(len(sumseq))
    
    @staticmethod
    def ci_normal(mean: float, std: float, level=0.95) -> tuple:
        """
        Calculates a confidence intervall for a given level
        mean: mean value
        sde: standard deviation
        level: CI level with default 95 %
        """
        alpha = 1 - level
        z_score = st.norm.ppf(1 - alpha / 2)
        upper = mean + z_score*std
        lower = mean- z_score*std
        return (lower, upper)