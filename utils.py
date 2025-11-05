import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def compute_convergence_prices(pricer):
    N_vals = np.logspace(2, 5, 50, dtype=int)
    prices = []
    std_errors = []
    for N in N_vals:
        price, se = pricer.call_price_european(num_simulations=N)
        prices.append(price)
        std_errors.append(se)
    return prices, std_errors, N_vals


def plot_convergence(prices, N_vals, bs_benchmark):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hlines(bs_benchmark, xmin=min(N_vals), xmax=max(N_vals), linestyle='--', linewidth=2, color='k', label='Black-Scholes')
    ax.plot(N_vals, prices, color='C0', label='Monte Carlo')
    ax.set_xlabel('Number of Simulated Paths (N, log scale)')
    ax.set_ylabel('Call Price')
    ax.set_xscale('log')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    ax.set_title('Convergence of Monte Carlo to Black-Scholes')

    ax.grid()
    ax.legend()
    plt.show()


def plot_standard_error(std_errors, N_vals):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.loglog(N_vals, std_errors, color='C0', label='Monte Carlo')
    ax.set_xlabel('Number of Simulated Paths (N, log scale)')
    ax.set_ylabel('Standard Error (SE, log scale)')
    ax.set_ylim(1e-1, 1e1)
    ax.set_title('Standard Error of Monte Carlo Estimates')

    coeffs = np.polyfit(np.log(N_vals), np.log(std_errors), 1)
    fit_line = np.exp(coeffs[1]) * N_vals ** coeffs[0]
    ax.loglog(N_vals, fit_line, ':', color='r', label=fr'Fit Line Slope: {coeffs[0]:.3f}')

    ax.grid()
    ax.legend()
    plt.show()


def plot_standard_error_comparison(std_errors_mc, std_errors_anti, N_vals):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.loglog(N_vals, std_errors_mc, label='Monte Carlo')
    ax.loglog(N_vals, std_errors_anti, label='Antithetic Monte Carlo')
    ax.set_xlabel('Number of Simulated Paths (N, log scale)')
    ax.set_ylabel('Standard Error (SE, log scale)')
    ax.set_ylim(1e-1, 1e1)
    ax.set_title('Standard Error of Monte Carlo vs Antithetic Monte Carlo')

    ax.legend()
    ax.grid()
    plt.show()


def plot_up_and_out_call(paths, time, hit, barrier):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hlines(barrier, xmin=min(time), xmax=max(time), linestyle='--', linewidth=2, color='r', label='Barrier B = 320')

    for i in range(paths.shape[1]):
        if hit[i]:
            ax.plot(time, paths[:, i], color='grey', alpha=0.5, label='Knocked-out Paths')
        else:
            ax.plot(time, paths[:, i], color='C0', label='Active Paths')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Stock Price')
    ax.set_xlim(time[0], time[-1])
    ax.set_title('Up-and-Out Barrier Call')

    ax.grid(alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
    plt.show()