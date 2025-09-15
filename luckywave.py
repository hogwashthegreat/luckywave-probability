import argparse
import numpy as np

def run_sim(N=100000000, plot=True, bins=100):
    rng = np.random.default_rng()

    # Draw N priors x ~ Uniform(0,1)
    x = rng.random(N, dtype=np.float64)

    # Simulate just the FIRST trial for each experiment
    first_success = rng.random(N) < x

    # Condition on first-trial success
    x_given_success = x[first_success]
    n_succ = int(first_success.sum())

    if n_succ == 0:
        print("No first-trial successes")
        return

    est = float(x_given_success.mean())
    se = float(x_given_success.std(ddof=1) / np.sqrt(n_succ))
    theoretical = 2.0 / 3.0

    print(f"N experiments: {N:,}")
    print(f"First-trial successes: {n_succ:,} ({n_succ/N:.3%})")
    print(f"Simulated E[x | success] ≈ {est:.6f}")
    print(f"Standard error of mean ≈ {se:.6f}")
    print(f"Theoretical value = 2/3 = {theoretical:.7f}")
    print(f"Difference (sim - theory) = {est - theoretical:.6f}")
    print(f"Implied expected successes in 1,000,000 future trials ≈ {est*1_000_000:,.0f}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            xs = np.linspace(0.0, 1.0, 400)
            pdf = 2.0 * xs  # Beta(2,1) density

            plt.figure(figsize=(7,5))
            plt.hist(x_given_success, bins=bins, density=False, weights=np.ones(x_given_success.size, dtype=float)/ x_given_success.size, alpha=0.6,
                     label="Empirical probability per bin")
            #plt.plot(xs, pdf, linewidth=2, label="Theoretical Beta(2,1) pdf = 2x")
            plt.xlabel("x")
            plt.ylabel("Probability (per bin)")
            plt.title("Posterior of x given FIRST trial success")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(Plotting skipped due to: {e})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num-experiments", type=int, default=100_000_000,
                   help="number of experiments to simulate (default: 100,000,000)")

    args = p.parse_args()

    run_sim(N=args.num_experiments)

if __name__ == "__main__":
    main()
