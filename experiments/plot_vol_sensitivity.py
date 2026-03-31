import numpy as np
import matplotlib.pyplot as plt
from option_pricer.black_scholes import BlackScholes

sigmas = np.linspace(0.05, 0.6, 100)
prices = [
    BlackScholes.black_scholes_price(100, 100, 1.0, 0.03, s, "call")
    for s in sigmas
]

plt.figure(figsize=(8, 5))
plt.plot(sigmas, prices)
plt.xlabel("Volatility")
plt.ylabel("Call price")
plt.title("Option price sensitivity to volatility")
plt.tight_layout()
plt.savefig("figures/vol_sensitivity.png", dpi=200)