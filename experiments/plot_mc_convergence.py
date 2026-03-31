import matplotlib.pyplot as plt
from option_pricer.black_scholes import BlackScholes
from option_pricer.mc import MonteCarlo

S0, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.2
n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]

bs = BlackScholes.black_scholes_price(S0, K, T, r, sigma, "call")
mc_prices = []

for n in n_values:
    price, _ = MonteCarlo.price_eu_option(
        n=n, S0=S0, K=K, T=T, rf=r, sigma=sigma,
        option_type="call", method="antithetic", seed=42
    )
    mc_prices.append(price)

plt.figure(figsize=(8, 5))
plt.plot(n_values, mc_prices, marker="o", label="Monte Carlo")
plt.axhline(bs, color="red", linestyle="--", label="Black-Scholes")
plt.xscale("log")
plt.xlabel("Number of paths")
plt.ylabel("Option price")
plt.title("Monte Carlo convergence to Black-Scholes")
plt.legend()
plt.tight_layout()
plt.savefig("figures/mc_convergence.png", dpi=200)