import matplotlib.pyplot as plt
from option_pricer.black_scholes import BlackScholes
from option_pricer.mc import MonteCarlo

S0, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]

bs = BlackScholes.black_scholes_price(S0, K, T, r, sigma, "call")
errors = []

for n in n_values:
    price, _ = MonteCarlo.price_eu_option(
        n=n, S0=S0, K=K, T=T, rf=r, sigma=sigma,
        option_type="call", method="antithetic", seed=42
    )
    errors.append(abs(price - bs))

plt.figure(figsize=(8, 5))
plt.plot(n_values, errors, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of paths")
plt.ylabel("Absolute error")
plt.title("Monte Carlo absolute pricing error")
plt.tight_layout()
plt.savefig("figures/mc_error.png", dpi=200)