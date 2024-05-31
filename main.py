import numpy as np
import pandas as pd
import wrds
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

# Define the Excel file names
excel_file = 'm_and_a_data.xlsx'
treasury_file = '1-Year Treasury Bill Rate.xls'


# Function to retrieve data from WRDS and save to Excel
def retrieve_and_save_data(sample_percentage=100):
    # Connect to WRDS
    conn = wrds.Connection(wrds_username='hugolmn')

    # Step 1: Retrieve all `gvkey` values from the company table
    gvkey_query = """
        SELECT gvkey, conm
        FROM comp.company
    """
    all_gvkey_data = conn.raw_sql(gvkey_query)
    total_companies = len(all_gvkey_data)

    # Calculate the sample size based on the percentage
    sample_size = int((sample_percentage / 100) * total_companies)

    # Step 2: Randomly select a sample of companies
    sample_gvkey_data = all_gvkey_data.sample(n=sample_size, random_state=1)

    # Extract company names and escape single quotes
    company_names = [name.replace("'", "''") for name in sample_gvkey_data['conm'].tolist()]

    # Step 3: Retrieve `permno` values for the sample companies using crsp.msenames
    permno_query = f"""
        SELECT DISTINCT permno, comnam
        FROM crsp.msenames
        WHERE comnam IN ({', '.join("'" + str(name) + "'" for name in company_names)})
    """
    permno_data = conn.raw_sql(permno_query)

    # Join with the sample_gvkey_data to get gvkey
    merged_data = pd.merge(sample_gvkey_data, permno_data, left_on='conm', right_on='comnam')

    # Extract permno values
    permnos = merged_data['permno'].tolist()

    # Extract gvkey values
    gvkeys = merged_data['gvkey'].tolist()

    # Step 4: Retrieve financial data for the sample and between 2015 and 2023
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

    # Calculate Free Cash Flow (FCF)
    fcf_components['fcf'] = fcf_components['oancf'] - fcf_components['capx']

    # Step 5: Retrieve daily stock returns for these companies between 2015 and 2023
    returns_query = f"""
        SELECT permno, date, ret
        FROM crsp.dsf
        WHERE permno IN ({', '.join("'" + str(permno) + "'" for permno in permnos)})
        AND date BETWEEN '2015-01-01' AND '2023-12-31'
    """
    returns_data = conn.raw_sql(returns_query)

    # Step 6: Calculate Volatility for Each Year
    returns_data['date'] = pd.to_datetime(returns_data['date'])
    returns_data['year'] = returns_data['date'].dt.year

    # Calculate annual volatility for each stock
    volatility_data = returns_data.groupby(['permno', 'year'], group_keys=False).apply(
        lambda x: pd.Series({'volatility': np.std(x['ret'].dropna()) * np.sqrt(252)})
    ).reset_index()

    # Rename 'year' to 'fyear' for merging
    volatility_data = volatility_data.rename(columns={'year': 'fyear'})

    # Step 7: Map `permno` to `gvkey`
    permno_to_gvkey = dict(zip(merged_data['permno'], merged_data['gvkey']))

    # Add `gvkey` to volatility data
    volatility_data['gvkey'] = volatility_data['permno'].map(permno_to_gvkey)

    # Merge FCF data with Volatility data
    final_data = pd.merge(fcf_components, volatility_data, on=['gvkey', 'fyear'], how='left')

    # Step 8: Retrieve Treasury Bill Rates from the file and match with the data
    # Step 8: Retrieve Treasury Bill Rates from the file and match with the data
    treasury_data = pd.read_excel(treasury_file)
    treasury_data['date'] = pd.to_datetime(treasury_data['date'])
    treasury_data['year'] = treasury_data['date'].dt.year
    treasury_data['rate'] = treasury_data['rate'].str.replace(',', '.').astype(float) / 100

    # Map each company's year to the corresponding treasury rate
    final_data['fyear'] = final_data['fyear'].astype(int)
    final_data = pd.merge(final_data, treasury_data[['year', 'rate']], left_on='fyear', right_on='year', how='left')
    final_data.rename(columns={'rate': 'average_interest_rate'}, inplace=True)

    # Drop unnecessary columns
    final_data.drop(columns=['year'], inplace=True)

    # Drop rows with NaN values
    final_data.dropna(inplace=True)

    # Filter out companies with insufficient data
    sufficient_data_threshold = 3  # Minimum number of years with data
    company_data_counts = final_data['gvkey'].value_counts()
    sufficient_data_gvkeys = company_data_counts[company_data_counts >= sufficient_data_threshold].index.tolist()
    final_data = final_data[final_data['gvkey'].isin(sufficient_data_gvkeys)]

    # Save data to Excel
    with pd.ExcelWriter(excel_file) as writer:
        final_data.to_excel(writer, sheet_name='Final Data', index=False)
        treasury_data.to_excel(writer, sheet_name='Treasury Rates', index=False)
        merged_data.to_excel(writer, sheet_name='Merged Data', index=False)

    return final_data


# Check if the Excel file exists and contains data
if os.path.exists(excel_file):
    # Load data from Excel
    final_data = pd.read_excel(excel_file, sheet_name='Final Data')
else:
    final_data = retrieve_and_save_data(sample_percentage=100)

# Ensure we drop any remaining rows with NaNs
final_data.dropna(inplace=True)

# Aggregate data by `gvkey`
aggregated_data = final_data.groupby('gvkey').agg({
    'fcf': lambda x: list(x),
    'volatility': 'mean',
    'average_interest_rate': 'mean'
}).reset_index()


def monte_carlo_simulation(company_data, num_simulations):
    fcf_values = np.array(company_data['fcf'].iloc[0])
    mean_fcf = np.mean(fcf_values)
    std_fcf = np.std(fcf_values)
    if std_fcf == 0 or np.isnan(std_fcf):
        return np.nan

    time_horizon = len(fcf_values)
    dt = 1  # Time step

    # Simulate Free Cash Flows using Geometric Brownian Motion
    simulated_fcf = np.zeros((num_simulations, time_horizon))
    simulated_fcf[:, 0] = fcf_values[-1]
    for t in range(1, time_horizon):
        dW = np.random.normal(0, np.sqrt(dt), size=num_simulations)
        drift = (mean_fcf - 0.5 * std_fcf ** 2) * dt
        diffusion = std_fcf * dW
        simulated_fcf[:, t] = simulated_fcf[:, t - 1] * np.exp(drift + diffusion)



    # Discount the simulated FCFs using the appropriate WACC
    market_risk_premium = 0.05
    average_interest_rate = company_data['average_interest_rate'].iloc[0]
    cost_of_equity = average_interest_rate + market_risk_premium
    cost_of_debt = average_interest_rate
    equity_weight = 0.6
    debt_weight = 0.4
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)

    # Debug: print WACC and discount factors
    print("WACC:", wacc)
    discount_factors = np.exp(-wacc * np.arange(1, time_horizon + 1))
    print("Discount factors:", discount_factors)

    discounted_fcf = simulated_fcf * discount_factors
    company_valuation = np.mean(np.sum(discounted_fcf, axis=1))

    # Debug: print final discounted FCF and valuation
    print("Discounted FCF:", discounted_fcf)
    print("Company Valuation:", company_valuation)

    return company_valuation



def jump_diffusion_simulation(company_data, num_simulations, jump_intensity, jump_mean, jump_std):
    fcf_values = np.array(company_data['fcf'].iloc[0])  # No scaling
    mean_fcf = np.mean(fcf_values)
    std_fcf = np.std(fcf_values)
    if std_fcf == 0 or np.isnan(std_fcf):
        return np.nan

    time_horizon = len(fcf_values)
    dt = 1 / 252

    # Simulate stock prices using GBM with jumps
    mean_vol = company_data['volatility'].iloc[0] / np.sqrt(252)
    std_vol = 0.01
    simulated_vol = np.random.lognormal(np.log(mean_vol), std_vol * 1.2, num_simulations)

    stock_prices = np.zeros((num_simulations, time_horizon))
    stock_prices[:, 0] = 1
    for t in range(1, time_horizon):
        # GBM components
        drift = (company_data['average_interest_rate'].iloc[0] - 0.5 * simulated_vol ** 2) * dt
        diffusion = simulated_vol * np.sqrt(dt) * np.random.standard_normal((num_simulations,))

        # Jump components
        jumps = np.random.poisson(jump_intensity * dt, num_simulations)
        jump_sizes = np.random.normal(jump_mean, jump_std, num_simulations)
        jump_effect = (np.exp(jump_sizes) - 1) * jumps

        stock_prices[:, t] = stock_prices[:, t - 1] * np.exp(drift + diffusion + jump_effect)

    # Calculate discounted FCFs and company valuation
    market_risk_premium = 0.05
    cost_of_equity = company_data['average_interest_rate'].iloc[0] + simulated_vol * market_risk_premium
    cost_of_debt = company_data['average_interest_rate'].iloc[0]
    equity_weight = 0.6
    debt_weight = 0.4
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)

    discount_factors = np.exp(-np.log(1 + wacc[:, None]) * np.arange(1, time_horizon + 1))
    discounted_fcf = np.sum(fcf_values * stock_prices * discount_factors, axis=1)
    company_valuation = np.sum(discounted_fcf)
    return company_valuation


# Number of simulations
num_simulations = 10000

# Perform Monte Carlo Simulation for each company
basic_valuations = []
jump_diffusion_valuations = []
jump_intensity = 0.1  # Average number of jumps per year
jump_mean = 0.05  # Mean of the jump size
jump_std = 0.1  # Standard deviation of the jump size

for gvkey in aggregated_data['gvkey']:
    company_data = aggregated_data[aggregated_data['gvkey'] == gvkey]
    if not company_data.empty:
        basic_valuation = monte_carlo_simulation(company_data, num_simulations)
        basic_valuations.append(basic_valuation)
        jump_diffusion_valuation = jump_diffusion_simulation(company_data, num_simulations, jump_intensity, jump_mean,
                                                             jump_std)
        jump_diffusion_valuations.append(jump_diffusion_valuation)

# Remove inf and NaN values
basic_valuations = np.array(basic_valuations)
basic_valuations = basic_valuations[np.isfinite(basic_valuations)]
jump_diffusion_valuations = np.array(jump_diffusion_valuations)
jump_diffusion_valuations = jump_diffusion_valuations[np.isfinite(jump_diffusion_valuations)]

# Compare Results
print(f'Basic Model - Average Valuation: {np.mean(basic_valuations)}')
print(f'Basic Model - Standard Deviation of Valuations: {np.std(basic_valuations)}')
print(f'Basic Model - 95% Confidence Interval of Valuations: {np.percentile(basic_valuations, [2.5, 97.5])}')

print(f'Jump-Diffusion Model - Average Valuation: {np.mean(jump_diffusion_valuations)}')
print(f'Jump-Diffusion Model - Standard Deviation of Valuations: {np.std(jump_diffusion_valuations)}')
print(
    f'Jump-Diffusion Model - 95% Confidence Interval of Valuations: {np.percentile(jump_diffusion_valuations, [2.5, 97.5])}')

print(f'Highest Valuation (Basic Model): {np.max(basic_valuations)}')
print(f'Lowest Valuation (Basic Model): {np.min(basic_valuations)}')
print(f'Highest Valuation (Jump-Diffusion Model): {np.max(jump_diffusion_valuations)}')
print(f'Lowest Valuation (Jump-Diffusion Model): {np.min(jump_diffusion_valuations)}')

# Identify factors influencing highest and lowest valuations
highest_valuation_gvkey_basic = aggregated_data['gvkey'][np.argmax(basic_valuations)]
lowest_valuation_gvkey_basic = aggregated_data['gvkey'][np.argmin(basic_valuations)]
highest_valuation_gvkey_jump_diffusion = aggregated_data['gvkey'][np.argmax(jump_diffusion_valuations)]
lowest_valuation_gvkey_jump_diffusion = aggregated_data['gvkey'][np.argmin(jump_diffusion_valuations)]

highest_valuation_data_basic = final_data[final_data['gvkey'] == highest_valuation_gvkey_basic]
lowest_valuation_data_basic = final_data[final_data['gvkey'] == lowest_valuation_gvkey_basic]
highest_valuation_data_jump_diffusion = final_data[final_data['gvkey'] == highest_valuation_gvkey_jump_diffusion]
lowest_valuation_data_jump_diffusion = final_data[final_data['gvkey'] == lowest_valuation_gvkey_jump_diffusion]

print("Factors Influencing Highest Valuation (Basic Model):")
print(highest_valuation_data_basic.describe())

print("Factors Influencing Lowest Valuation (Basic Model):")
print(lowest_valuation_data_basic.describe())

print("Factors Influencing Highest Valuation (Jump-Diffusion Model):")
print(highest_valuation_data_jump_diffusion.describe())

print("Factors Influencing Lowest Valuation (Jump-Diffusion Model):")
print(lowest_valuation_data_jump_diffusion.describe())

# Visualize Results
plt.hist(np.log1p(basic_valuations), bins=50, alpha=0.5, label='Basic Model')
plt.hist(np.log1p(jump_diffusion_valuations), bins=50, alpha=0.5, label='Jump-Diffusion Model')
plt.title('Histogram of Log-Transformed Valuations from Monte Carlo Simulations')
plt.xlabel('Log(Company Valuation)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.hist(basic_valuations, bins=50, alpha=0.5, label='Basic Model')
plt.hist(jump_diffusion_valuations, bins=50, alpha=0.5, label='Jump-Diffusion Model')
plt.title('Histogram of Valuations from Monte Carlo Simulations')
plt.xlabel('Company Valuation')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Sensitivity Analysis
def sensitivity_analysis(company_data, num_simulations, parameter, param_range):
    valuations = []
    for value in param_range:
        if parameter == 'volatility':
            original_vol = company_data['volatility'].iloc[0]
            company_data.loc[company_data.index[0], 'volatility'] = value
            valuation = monte_carlo_simulation(company_data, num_simulations)
            company_data.loc[company_data.index[0], 'volatility'] = original_vol  # Restore original value
        elif parameter == 'jump_intensity':
            valuation = jump_diffusion_simulation(company_data, num_simulations, value, jump_mean, jump_std)
        valuations.append(valuation)
    return valuations


# Select a company for sensitivity analysis
company_data = aggregated_data[aggregated_data['gvkey'] == aggregated_data['gvkey'].iloc[0]]

volatility_range = np.linspace(0.1, 0.5, 10)
jump_intensity_range = np.linspace(0, 0.5, 10)

volatility_sensitivity = sensitivity_analysis(company_data, num_simulations, 'volatility', volatility_range)
jump_intensity_sensitivity = sensitivity_analysis(company_data, num_simulations, 'jump_intensity', jump_intensity_range)

plt.plot(volatility_range, volatility_sensitivity, label='Volatility Sensitivity')
plt.xlabel('Volatility')
plt.ylabel('Company Valuation')
plt.legend()
plt.show()

plt.plot(jump_intensity_range, jump_intensity_sensitivity, label='Jump Intensity Sensitivity')
plt.xlabel('Jump Intensity')
plt.ylabel('Company Valuation')
plt.legend()
plt.show()

# Practical implications for M&A
print("Practical Implications for M&A:")
print(
    "The models help in better decision-making by providing a range of potential valuations under different scenarios.")
print(
    "Considering sudden changes in firm value (jumps) during M&A evaluations and negotiations is crucial for more accurate assessments.")
