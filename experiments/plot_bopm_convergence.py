import matplotlib.pyplot as plt
from option_pricer.bopm import BinomialModel

n_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
prices = []

for n in n_values:
    p = BinomialModel.price_american_option(
        n=n, S0=100, K=100, T=1.0, r=0.03, sigma=0.2, option_type="put"
    )
    prices.append(p)

plt.figure(figsize=(8, 5))
plt.plot(n_values, prices, marker="o")
plt.xscale("log")
plt.xlabel("Number of tree steps")
plt.ylabel("American put price")
plt.title("Binomial model convergence")
plt.tight_layout()
plt.savefig("figures/bopm_convergence.png", dpi=200)