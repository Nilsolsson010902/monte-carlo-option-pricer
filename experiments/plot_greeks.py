import numpy as np
import matplotlib.pyplot as plt
from option_pricer.black_scholes import BlackScholes

S_values = np.linspace(60, 140, 200)
delta = [BlackScholes.delta(S, 100, 1.0, 0.03, 0.2, "call") for S in S_values]
gamma = [BlackScholes.gamma(S, 100, 1.0, 0.03, 0.2) for S in S_values]

plt.figure(figsize=(8, 5))
plt.plot(S_values, delta, label="Delta")
plt.plot(S_values, gamma, label="Gamma")
plt.xlabel("Spot price")
plt.ylabel("Sensitivity")
plt.title("Black-Scholes Delta and Gamma vs Spot Price")
plt.axvline(100, linestyle="--", color="gray", alpha=0.7, label="Strike")
plt.legend()
plt.tight_layout()
plt.savefig("figures/greeks.png", dpi=200)