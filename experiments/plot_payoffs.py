import numpy as np
import matplotlib.pyplot as plt

S = np.linspace(50, 150, 400)
K = 100

call = np.maximum(S - K, 0)
put = np.maximum(K - S, 0)

plt.figure(figsize=(8, 5))
plt.plot(S, call, label="Call payoff")
plt.plot(S, put, label="Put payoff")
plt.axvline(K, color="gray", linestyle="--", alpha=0.7)
plt.xlabel("Terminal stock price")
plt.ylabel("Payoff")
plt.title("European option payoffs")
plt.legend()
plt.tight_layout()
plt.savefig("figures/payoff.png", dpi=200)