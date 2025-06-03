# Valuation Simulation Toolkit – README

## 1. Project Overview

This repository accompanies my academic paper on **corporate valuation methods in the context of mergers and acquisitions (M\&A)**.  Its goal is to provide a fully reproducible pipeline—from raw market data to publish‑ready figures—that lets researchers and practitioners compare **stochastic valuation techniques** with **machine‑learning forecasts**.

### 1.1 Motivation

When a takeover bid is announced, equity prices often diverge from fundamental or deal‑implied values.  Standard discounted‑cash‑flow (DCF) models struggle to capture the complex mixture of market sentiment, execution risk, and macro shocks.  To tackle this, the toolkit:

* Simulates thousands (or millions) of future price paths using **Geometric Brownian Motion (GBM)** and **jump‑diffusion processes**, generating a distribution of firm values rather than a single point estimate.
* Trains **supervised learning models** (currently a Random‑Forest Regressor) on historical deal data to predict post‑announcement returns and completion probabilities.
* Benchmarks the two approaches side‑by‑side so that the relative explanatory power of market micro‑structure noise versus fundamental drivers can be quantified.

### 1.2 Components

| Script                              | Purpose                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------- |
| `monte_carlo_pipeline.py`           | End‑to‑end GBM valuation: data pull → simulation → statistical summary.          |
| `valuation_no_brownian.py`          | Jump‑diffusion variant without a Brownian component (stress‑testing tail risk).  |
| `ml_and_simulation_pipeline.py`     | Combined ML + simulation workflow with automated hyper‑parameter tuning.         |
| `activision_microsoft_valuation.py` | Stand‑alone case study of Microsoft’s 2022 bid for Activision Blizzard.          |
| `valuation_visualisation.py`        | Helper for overlaying ML vs simulation histograms and cumulative‑density plots.  |
| `treasury_1y_bill_rate.xls`         | Daily risk‑free rates (FRED) used for discounting cash flows.                    |
| `m_and_a_data.xlsx`                 | Auto‑generated workbook storing cleaned feature matrices and simulation outputs. |

All scripts share a common helper module (`utils/`) that handles WRDS authentication, logging, and figure styling.

### 1.3 Data Flow in a Nutshell

```
┌──────────────┐     fetch      ┌──────────────┐   preprocess  ┌───────────────────────┐
│  WRDS/CRSP   │──────────────▶│  pandas DF    │──────────────▶│ Monte‑Carlo Simulator │
└──────────────┘                └──────────────┘               └─────────┬─────────────┘
                                                                      write ↓
                                                               ┌─────────────────┐
                                                               │   outputs/      │
                                                               │  • charts (.png)│
                                                               │  • tables (.csv)│
                                                               └─────────────────┘
```

The ML branch follows a parallel path, producing feature‑importance plots and out‑of‑sample error metrics for immediate comparison with the stochastic results.

### 1.4 Extensibility

* **Additional factors**: Swap in alternative factor models (Five‑Factor, Q‑Factor, etc.) by editing `data_loader.py`.
* **Different ML models**: Any scikit‑learn regressor exposes the same fit/predict interface; drop‑in replacement takes one line.
* **New case studies**: Point the scripts to a different ticker pair and update deal terms in `config.yaml`.

## 2. How to Run the Project

### 2.1 Requirements

* Python 3.10 or newer.
* A WRDS account with access to CRSP and Compustat.
* Packages:

  ```bash
  pip install numpy pandas scipy matplotlib scikit-learn wrds openpyxl
  ```

### 2.2 One‑Minute Setup

```bash
# Clone the repository
git clone <repo_url>
cd <repo_name>

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set WRDS credentials (Unix/macOS example)
export WRDS_USERNAME="your_wrds_username"
export WRDS_PASSWORD="your_wrds_password"

# Reproduce the main results
python monte_carlo_pipeline.py           # Figures 2–3 in the paper
python ml_and_simulation_pipeline.py     # Figures 4–5
python activision_microsoft_valuation.py # Appendix A case study

# All generated artefacts appear in the outputs/ directory
```

---

© 2025 Hugo L. – Licensed under the MIT License.
