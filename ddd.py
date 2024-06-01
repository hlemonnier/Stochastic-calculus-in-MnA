import numpy as np
from matplotlib import pyplot as plt


def visualize_comparison(mc_valuations, ml_valuation, ml_valuations):
    # Print debug information for values being plotted
    print(f"ML Valuations: min={np.min(ml_valuations)}, max={np.max(ml_valuations)}, mean={np.mean(ml_valuations)}")
    print(f"MC Valuations: min={np.min(mc_valuations)}, max={np.max(mc_valuations)}, mean={np.mean(mc_valuations)}")
    print(f"ML Predicted Valuation: {ml_valuation}")

    plt.figure(figsize=(14, 8))

    # Plot ML Valuations first with more transparency
    n, bins, patches = plt.hist(ml_valuations, bins=50, alpha=0.4, label='ML Valuations', color='orange', edgecolor='black', zorder=2)
    print(f"ML Histogram: bins={bins}")

    # Plot Monte Carlo Valuations behind ML Valuations
    n, bins, patches = plt.hist(mc_valuations, bins=bins, alpha=0.6, label='Monte Carlo Valuations', color='blue', edgecolor='black', zorder=1)
    print(f"MC Histogram: bins={bins}")

    # Plot the ML Predicted Valuation as a vertical line
    plt.axvline(ml_valuation, color='r', linestyle='--', linewidth=2, label='ML Predicted Valuation', zorder=3)

    plt.title('Comparison of Monte Carlo and Machine Learning Valuations')
    plt.xlabel('Valuation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

# Dummy data for demonstration purposes
mc_valuations = np.random.exponential(scale=50000, size=10000)
ml_valuations = np.random.normal(loc=47000, scale=500, size=10000)
ml_valuation = np.mean(ml_valuations)

visualize_comparison(mc_valuations, ml_valuation, ml_valuations)