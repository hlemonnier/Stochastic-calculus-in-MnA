import numpy as np
import pandas as pd
import wrds
import matplotlib.pyplot as plt

# Connect to WRDS
conn = wrds.Connection(wrds_username='hugolmn')

# Step 1: Retrieve `gvkey` values for Microsoft and Activision
company_names = ["MICROSOFT CORP", "ACTIVISION BLIZZARD INC"]

gvkey_query = f"""
    SELECT gvkey, conm
    FROM comp.company
    WHERE conm IN ({', '.join("'" + name + "'" for name in company_names)})
"""

gvkey_data = conn.raw_sql(gvkey_query)
print("GVKEY Data:")
print(gvkey_data)

# Extract gvkey values
gvkeys = gvkey_data['gvkey'].tolist()
print("Extracted GVKEYs:", gvkeys)

# Step 2: Retrieve financial data for these `gvkey` values and between 2015 and 2023
fcf_query = f"""
    SELECT gvkey, datadate, fyear, oancf, capx
    FROM comp.funda
    WHERE gvkey IN ({', '.join("'" + str(gvkey) + "'" for gvkey in gvkeys)})
    AND indfmt='INDL' 
    AND datafmt='STD' 
    AND popsrc='D' 
    AND consol='C'
    AND fyear BETWEEN 2015 AND 2023
"""

fcf_components = conn.raw_sql(fcf_query)
print("FCF Components Data:")
print(fcf_components.head())

# Calculate Free Cash Flow (FCF)
fcf_components['fcf'] = fcf_components['oancf'] - fcf_components['capx']

# Display the first few rows to verify the calculation
print("FCF Data:")
print(fcf_components.head())

# Step 3: Retrieve `permno` values for Microsoft and Activision
ticker_symbols = ["MSFT", "ATVI"]

permno_query = f"""
    SELECT permno, ticker
    FROM crsp.stocknames
    WHERE ticker IN ({', '.join("'" + symbol + "'" for symbol in ticker_symbols)})
"""

permno_data = conn.raw_sql(permno_query)
print("Permno Data:")
print(permno_data)

# Extract permno values
permnos = permno_data['permno'].tolist()
print("Extracted Permnos:", permnos)

# Step 4: Retrieve daily stock returns for these companies between 2015 and 2023
returns_query = f"""
    SELECT permno, date, ret
    FROM crsp.dsf
    WHERE permno IN ({', '.join("'" + str(permno) + "'" for permno in permnos)})
    AND date BETWEEN '2015-01-01' AND '2023-12-31'
"""

returns_data = conn.raw_sql(returns_query)
print("Returns Data:")
print(returns_data.head())

# Step 5: Calculate Volatility for Each Year
returns_data['date'] = pd.to_datetime(returns_data['date'])
returns_data['year'] = returns_data['date'].dt.year

# Calculate annual volatility for each stock
volatility_data = returns_data.groupby(['permno', 'year'], group_keys=False).apply(
    lambda x: pd.Series({'volatility': np.std(x['ret'].dropna()) * np.sqrt(252)})
).reset_index()

print("Annual Volatility Data:")
print(volatility_data.head())

# Rename 'year' to 'fyear' for merging
volatility_data = volatility_data.rename(columns={'year': 'fyear'})

# Step 6: Map `permno` to `gvkey`
permno_to_gvkey = dict(zip(permno_data['permno'], gvkey_data['gvkey']))

# Add `gvkey` to volatility data
volatility_data['gvkey'] = volatility_data['permno'].map(permno_to_gvkey)
print("Volatility Data with GVKEY:")
print(volatility_data.head())

# Merge FCF data with Volatility data
final_data = pd.merge(fcf_components, volatility_data, on=['gvkey', 'fyear'], how='left')


# Step 8: Retrieve Interest Rate Data from the 'factors_monthly' table
interest_rate_query = f"""
    SELECT date, rf as rate
    FROM ff.factors_monthly
    WHERE date BETWEEN '2015-01-01' AND '2023-12-31'
"""

interest_rate_data = conn.raw_sql(interest_rate_query)
print("Interest Rate Data:")
print(interest_rate_data.head())

# Step 9: Calculate the Average Monthly Rate for Each Year and Annualize It
interest_rate_data['date'] = pd.to_datetime(interest_rate_data['date'])
interest_rate_data['year'] = interest_rate_data['date'].dt.year

# Calculate the average monthly rate for each year
avg_monthly_rate_per_year = interest_rate_data.groupby('year')['rate'].mean()

# Annualize the average monthly rate for each year
annualized_rate_per_year = ((1 + avg_monthly_rate_per_year) ** 12 - 1) * 100
print("Annualized Rate per Year:")
print(annualized_rate_per_year)

# Calculate the average annualized rate over the specified period
average_annualized_rate = annualized_rate_per_year.mean()
print(f"Average Annualized Interest Rate (2015-2023): {average_annualized_rate:.2f}%")

# Step 10: Add the average annualized rate to the final data
final_data['average_interest_rate'] = average_annualized_rate

# Display the final merged data with interest rate
print("Final Data with Interest Rate:")
print(final_data)

num_simulations=10
# Separate data for Microsoft and Activision
msft_data = final_data[final_data['gvkey'] == '012141']
atvi_data = final_data[final_data['gvkey'] == '180405']
# Check for NaN values in both datasets
print("Microsoft Data:")
print(msft_data.isna().sum())
print("Activision Data:")
print(atvi_data.isna().sum())

# Print the final_data to inspect the interest rates and other values
print("Final Data for Microsoft:")
print(msft_data[['fcf', 'volatility', 'average_interest_rate']].describe())
print("Final Data for Activision:")
print(atvi_data[['fcf', 'volatility', 'average_interest_rate']].describe())


# Function to perform Monte Carlo Simulation for a given company
def monte_carlo_simulation(company_data, num_simulations, average_annualized_rate):
    mean_fcf = company_data['fcf'].mean()
    std_fcf = company_data['fcf'].std()
    print(f"Mean FCF: {mean_fcf}, Std FCF: {std_fcf}")
    if std_fcf == 0 or np.isnan(std_fcf):
        print(f"Standard deviation of FCF is zero or NaN for company with gvkey {company_data['gvkey'].iloc[0]}")
        return (np.array([np.nan] * num_simulations),) * 3

    # Introduce synthetic variability to FCF
    simulated_fcf = np.random.normal(mean_fcf, std_fcf * 1.2, num_simulations)
    print(f"Simulated FCF: {simulated_fcf[:5]}")

    mean_ir = average_annualized_rate / 100  # Convert back to decimal for simulation
    simulated_ir = np.random.normal(mean_ir, mean_ir * 0.05, num_simulations)  # Introduce 5% variability
    print(f"Simulated IR: {simulated_ir[:5]}")

    mean_vol = company_data['volatility'].mean()
    std_vol = company_data['volatility'].std()
    print(f"Mean Volatility: {mean_vol}, Std Volatility: {std_vol}")
    if std_vol == 0 or np.isnan(std_vol):
        std_vol = 0.01  # Introduce a small variability

    simulated_vol = np.random.lognormal(np.log(mean_vol), std_vol * 1.2, num_simulations)  # Increase variability
    print(f"Simulated Volatility: {simulated_vol[:5]}")

    return simulated_fcf, simulated_ir, simulated_vol


# Perform Monte Carlo Simulation for Microsoft
print("Microsoft Monte Carlo Simulation:")
msft_sim_fcf, msft_sim_ir, msft_sim_vol = monte_carlo_simulation(msft_data, num_simulations, average_annualized_rate)

# Perform Monte Carlo Simulation for Activision
print("Activision Monte Carlo Simulation:")
atvi_sim_fcf, atvi_sim_ir, atvi_sim_vol = monte_carlo_simulation(atvi_data, num_simulations, average_annualized_rate)

# Combine FCFs
print("Combining FCFs:")
combined_fcf = msft_sim_fcf + atvi_sim_fcf
print(f"Combined FCF: {combined_fcf[:5]}")

# Calculate combined WACC
equity_weight = 0.6
debt_weight = 0.4
market_risk_premium = 0.05

combined_ir = (msft_sim_ir + atvi_sim_ir) / 2
combined_vol = (msft_sim_vol + atvi_sim_vol) / 2
print(f"Combined IR: {combined_ir[:5]}, Combined Volatility: {combined_vol[:5]}")

cost_of_equity = combined_ir + combined_vol * market_risk_premium
cost_of_debt = combined_ir
print(f"Cost of Equity: {cost_of_equity[:5]}, Cost of Debt: {cost_of_debt[:5]}")

combined_wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)
print(f"Combined WACC: {combined_wacc[:5]}")

# Discount Combined FCFs
discounted_combined_fcf = combined_fcf / (1 + combined_wacc)
print(f"Discounted Combined FCF: {discounted_combined_fcf[:5]}")
combined_valuation = np.sum(discounted_combined_fcf)
print(f"Average combined valuation from Monte Carlo simulations: {np.mean(combined_valuation)}")

# Analyze Results
if not np.isnan(combined_valuation).all():
    plt.hist(combined_valuation, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Histogram of Combined Valuations from Monte Carlo Simulations')
    plt.xlabel('Company Valuation')
    plt.ylabel('Frequency')
    plt.show()

    combined_valuation_mean = np.mean(combined_valuation)
    combined_valuation_std = np.std(combined_valuation)
    combined_valuation_95_ci = np.percentile(combined_valuation, [2.5, 97.5])

    print(f'Average Combined Valuation: {combined_valuation_mean}')
    print(f'Standard Deviation of Combined Valuations: {combined_valuation_std}')
    print(f'95% Confidence Interval of Combined Valuations: {combined_valuation_95_ci}')
else:
    print("All combined valuations are NaN.")
