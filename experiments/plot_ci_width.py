import matplotlib.pyplot as plt
from option_pricer.mc import MonteCarlo

S0, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
widths = []

for n in n_values:
    _, ci = MonteCarlo.price_eu_option(
        n=n, S0=S0, K=K, T=T, rf=r, sigma=sigma,
        option_type="call", method="antithetic", seed=42
    )
    widths.append(ci[1] - ci[0])

plt.figure(figsize=(8, 5))
plt.plot(n_values, widths, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of paths")
plt.ylabel("Confidence interval width")
plt.title("Monte Carlo confidence interval width")
plt.tight_layout()
plt.savefig("figures/ci_width.png", dpi=200)