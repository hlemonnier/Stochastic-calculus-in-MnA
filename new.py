import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import wrds
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# Define the Excel file names
excel_file = 'm_and_a_data.xlsx'
treasury_file = '1-Year Treasury Bill Rate.xls'


def retrieve_and_save_data(sample_percentage=100):
    # Connect to WRDS
    conn = wrds.Connection(wrds_username='hugolmn')

    # Step 1: Retrieve all `gvkey` values from the company table
    gvkey_query = """
        SELECT gvkey, conm
        FROM comp.company
    """
    all_gvkey_data = conn.raw_sql(gvkey_query)
    all_gvkey_data['gvkey'] = all_gvkey_data['gvkey'].astype(str)  # Convert to string
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

    # Step 4: Retrieve financial data for the sample and between 2008 and 2023
    fcf_query = f"""
        SELECT gvkey, datadate, fyear, oancf, capx, csho
        FROM comp.funda
        WHERE gvkey IN ({', '.join("'" + str(gvkey) + "'" for gvkey in gvkeys)})
        AND indfmt='INDL' 
        AND datafmt='STD' 
        AND popsrc='D' 
        AND consol='C'
        AND fyear BETWEEN 2008 AND 2024
    """
    fcf_components = conn.raw_sql(fcf_query)
    fcf_components['gvkey'] = fcf_components['gvkey'].astype(str)  # Convert to string

    # Calculate Free Cash Flow (FCF)
    fcf_components['fcf'] = fcf_components['oancf'] - fcf_components['capx']

    # Step 5: Retrieve daily stock returns for these companies between 2008 and 2023
    returns_query = f"""
        SELECT permno, date, ret, prc
        FROM crsp.dsf
        WHERE permno IN ({', '.join("'" + str(permno) + "'" for permno in permnos)})
        AND date BETWEEN '2008-01-01' AND '2024-12-31'
    """
    returns_data = conn.raw_sql(returns_query)

    # Step 6: Calculate Volatility and Average Price for Each Year
    returns_data['date'] = pd.to_datetime(returns_data['date'])
    returns_data['year'] = returns_data['date'].dt.year

    # Calculate annual volatility and average price for each stock
    aggregated_returns_data = returns_data.groupby(['permno', 'year'], group_keys=False).agg({
        'ret': lambda x: np.std(x.dropna()) * np.sqrt(252),
        'prc': 'mean'
    }).reset_index().rename(columns={'ret': 'volatility', 'prc': 'average_price'})

    # Rename 'year' to 'fyear' for merging
    aggregated_returns_data = aggregated_returns_data.rename(columns={'year': 'fyear'})

    # Step 7: Map `permno` to `gvkey`
    permno_to_gvkey = dict(zip(merged_data['permno'], merged_data['gvkey']))

    # Add `gvkey` to aggregated returns data
    aggregated_returns_data['gvkey'] = aggregated_returns_data['permno'].map(permno_to_gvkey)

    # Merge FCF data with Volatility and Average Price data
    final_data = pd.merge(fcf_components, aggregated_returns_data, on=['gvkey', 'fyear'], how='left')

    # Step 8: Retrieve Treasury Bill Rates from the file and match with the data
    treasury_data = pd.read_excel(treasury_file)
    treasury_data['date'] = pd.to_datetime(treasury_data['date'])
    treasury_data['year'] = treasury_data['date'].dt.year

    # Convert 'rate' column to string and replace commas with dots, then convert to float
    treasury_data['rate'] = treasury_data['rate'].astype(str).str.replace(',', '.').astype(float) / 100

    # Map each company's year to the corresponding treasury rate
    final_data['fyear'] = final_data['fyear'].astype(int)
    final_data = pd.merge(final_data, treasury_data[['year', 'rate']], left_on='fyear', right_on='year', how='left')
    final_data.rename(columns={'rate': 'average_interest_rate'}, inplace=True)

    # Drop unnecessary columns
    final_data.drop(columns=['year'], inplace=True)

    # Drop rows with NaN values
    final_data.dropna(inplace=True)

    # Filter out companies with insufficient data
    sufficient_data_threshold = 0  # Minimum number of years with data
    company_data_counts = final_data['gvkey'].value_counts()
    sufficient_data_gvkeys = company_data_counts[company_data_counts >= sufficient_data_threshold].index.tolist()
    final_data = final_data[final_data['gvkey'].isin(sufficient_data_gvkeys)]

    # Step 9: Create manual M&A events data
    ma_data = pd.DataFrame({
        'master_deal_no': [180405],  # gvkey for Activision Blizzard
        'dateann': ['2022-01-18'],
        'amanames': ['Microsoft Corporation'],
        'tmanames': ['Activision Blizzard, Inc.'],
        'entval': [np.nan],
        'eqval': [np.nan],
        'status': [np.nan]
    })
    ma_data['master_deal_no'] = ma_data['master_deal_no'].astype(str)
    ma_data['dateann'] = pd.to_datetime(ma_data['dateann'])
    ma_data['fyear'] = ma_data['dateann'].dt.year


    # Ensure gvkey is of type string in final_data
    final_data['gvkey'] = final_data['gvkey'].astype(str)

    # Merge M&A data with final dataset
    final_data_with_ma = pd.merge(final_data, ma_data, left_on=['gvkey', 'fyear'], right_on=['master_deal_no', 'fyear'],
                                  how='left')

    # Save data to Excel
    with pd.ExcelWriter(excel_file) as writer:
        final_data_with_ma.to_excel(writer, sheet_name='Final Data', index=False)
        treasury_data.to_excel(writer, sheet_name='Treasury Rates', index=False)
        ma_data.to_excel(writer, sheet_name='M&A Data', index=False)

    return final_data_with_ma


# Check if the Excel file exists and contains data
if os.path.exists(excel_file):
    # Load data from Excel
    final_data_with_ma = pd.read_excel(excel_file, sheet_name='Final Data')
else:
    final_data_with_ma = retrieve_and_save_data(sample_percentage=100)


# Aggregate data by `gvkey` and `fyear`
aggregated_data = final_data_with_ma.groupby(['gvkey', 'fyear']).agg({
    'fcf': 'mean',
    'volatility': 'mean',
    'average_price': 'mean',
    'average_interest_rate': 'mean',
    'csho': 'mean'  # Aggregate the number of outstanding shares
}).reset_index()


# Function to plot stock prices around M&A events for a target company
def plot_stock_prices_around_ma(data, target_gvkey):
    target_company_data = data[data['gvkey'] == target_gvkey]

    if not target_company_data.empty:
        plt.figure(figsize=(14, 8))

        # Plot the real average prices per year
        plt.plot(target_company_data['fyear'], target_company_data['average_price'], marker='o', linestyle='-',
                 color='blue', label='Real Average Price')

        # Highlight M&A events
        ma_events = target_company_data.dropna(subset=['dateann'])
        for idx, event in ma_events.iterrows():
            plt.axvline(x=event['fyear'], color='red', linestyle='--',
                        label=f"M&A Event: {event['amanames']} acquired {event['tmanames']} on {event['dateann'].date()}")

        plt.title(f'Real Prices and M&A Events for GVKEY: {target_gvkey}')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for GVKEY 1: {target_gvkey}")


# Function to generate and plot Brownian Motion paths for each company-year
def generate_brownian_motion(aggregated_data, target_gvkey, T=16.0, N=1000):
    dt = T / N  # Time step
    t = np.linspace(0, T, N)  # Time vector

    # Plot for all companies
    plt.figure(figsize=(14, 8))
    unique_gvkeys = aggregated_data['gvkey'].unique()

    for gvkey in unique_gvkeys:
        company_data = aggregated_data[aggregated_data['gvkey'] == gvkey]

        for i, row in company_data.iterrows():
            volatility = row['volatility']

            # Generate Brownian Motion path
            W = np.zeros(N)
            W[1:] = np.cumsum(np.sqrt(dt) * np.random.randn(N - 1) * volatility)

            plt.plot(t, W, label=f'GVKEY: {gvkey}')

    plt.title('Standard Brownian Motion for Different Companies')
    plt.xlabel('Time (t)')
    plt.ylabel('W(t)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

    # Plot specifically for target_gvkey
    target_company_data = aggregated_data[aggregated_data['gvkey'] == target_gvkey]

    if not target_company_data.empty:
        plt.figure(figsize=(14, 8))

        for i, row in target_company_data.iterrows():
            volatility = row['volatility']
            fyear = row['fyear']

            # Generate Brownian Motion path
            W = np.zeros(N)
            W[1:] = np.cumsum(np.sqrt(dt) * np.random.randn(N - 1) * volatility)

            plt.plot(t, W, label=f'Year: {fyear}')

        plt.title(f'Standard Brownian Motion for GVKEY: {target_gvkey}')
        plt.xlabel('Time (t)')
        plt.ylabel('W(t)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for GVKEY 2: {target_gvkey}")


# Generate and plot Geometric Brownian Motion paths for each company-year
def geometric_brownian_motion(S0, T, mu, sigma, N):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian motion
    return t, S


# Plotting real, Brownian Motion, and GBM simulated prices
def plot_real_vs_simulated_prices(aggregated_data, target_gvkey, T=16.0, N=1000):
    target_company_data = aggregated_data[aggregated_data['gvkey'] == target_gvkey]

    if not target_company_data.empty:
        plt.figure(figsize=(14, 8))

        # Plot the real average prices per year
        plt.plot(target_company_data['fyear'], target_company_data['average_price'], marker='o', linestyle='-',
                 color='blue', label='Real Average Price')

        # Generate and plot Brownian Motion paths
        for i, row in target_company_data.iterrows():
            volatility = row['volatility']
            initial_price = row['average_price']

            # Generate Brownian Motion path
            W = np.zeros(N)
            W[1:] = np.cumsum(np.sqrt(T / N) * np.random.randn(N - 1) * volatility)
            plt.plot(np.linspace(row['fyear'], row['fyear'] + T / len(target_company_data), N), initial_price + W,
                     linestyle='--', label=f'Brownian Motion {row["fyear"]}')

            # Generate Geometric Brownian Motion path
            mu = 0.1  # Assuming a constant drift, adjust based on actual data if needed
            t, S = geometric_brownian_motion(initial_price, T / len(target_company_data), mu, volatility, N)
            plt.plot(np.linspace(row['fyear'], row['fyear'] + T / len(target_company_data), N), S, linestyle=':',
                     label=f'GBM {row["fyear"]}')

        plt.title(f'Real, Brownian Motion, and GBM Simulated Prices for GVKEY: {target_gvkey}')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No data available for GVKEY 3: {target_gvkey}")


# Step 1: Impact of M&A on Stock Price Volatility
def calculate_volatility_impact(data, target_gvkey):
    ma_events = data[(data['gvkey'] == target_gvkey) & (~data['dateann'].isna())]
    if ma_events.empty:
        print(f"No M&A events found for GVKEY: {target_gvkey}")
        return

    volatilities = []
    for _, event in ma_events.iterrows():
        year = event['fyear']
        pre_ma_data = data[(data['gvkey'] == target_gvkey) & (data['fyear'] == year - 1)]
        post_ma_data = data[(data['gvkey'] == target_gvkey) & (data['fyear'] == year + 1)]

        if not pre_ma_data.empty and not post_ma_data.empty:
            pre_ma_volatility = pre_ma_data['volatility'].values[0]
            post_ma_volatility = post_ma_data['volatility'].values[0]
            volatilities.append((year, pre_ma_volatility, post_ma_volatility))

    if volatilities:
        df = pd.DataFrame(volatilities, columns=['Year', 'Pre_MA_Volatility', 'Post_MA_Volatility'])
        df.plot(x='Year', y=['Pre_MA_Volatility', 'Post_MA_Volatility'], kind='bar')
        plt.title(f'Impact of M&A on Stock Price Volatility for GVKEY: {target_gvkey}')
        plt.xlabel('Year')
        plt.ylabel('Volatility')
        plt.show()
    else:
        print(f"Not enough data to calculate volatility impact for GVKEY: {target_gvkey}")


# Step 3: Risk and Return Analysis
def calculate_risk_return(data, target_gvkey):
    post_ma_data = data[(data['gvkey'] == target_gvkey) & (~data['dateann'].isna())]

    if post_ma_data.empty:
        print(f"No post-M&A data found for GVKEY: {target_gvkey}")
        return

    returns = post_ma_data['average_price'].pct_change().dropna()
    expected_return = returns.mean()
    risk = returns.std()

    print(f"Expected Return for GVKEY {target_gvkey}: {expected_return:.2%}")
    print(f"Risk (Standard Deviation) for GVKEY {target_gvkey}: {risk:.2%}")


# Monte Carlo Simulation for company valuation using stock prices and outstanding shares
def monte_carlo_valuation_stock(data, target_gvkey, num_simulations=10000, T=5):
    target_company_data = data[data['gvkey'] == target_gvkey]

    if target_company_data.empty:
        print(f"No data available for GVKEY: {target_gvkey}")
        return

    # Extract relevant data
    initial_price = target_company_data['average_price'].values[-1]
    avg_volatility = target_company_data['volatility'].mean()
    avg_interest_rate = target_company_data['average_interest_rate'].mean()
    mu = avg_interest_rate  # Assuming drift is equal to the average interest rate
    outstanding_shares = target_company_data['csho'].values[-1]  # Number of shares outstanding

    # Define the simulation parameters
    dt = 1  # time step in years
    sim_prices = np.zeros((num_simulations, T))
    sim_valuations = np.zeros(num_simulations)
    sim_data = []

    for i in range(num_simulations):
        prices = [initial_price]
        for t in range(1, T):
            price_t = prices[-1] * np.exp((mu - 0.5 * avg_volatility ** 2) * dt + avg_volatility * np.sqrt(dt) * np.random.randn())
            prices.append(price_t)
        sim_prices[i, :] = prices
        sim_valuations[i] = prices[-1] * outstanding_shares  # Market capitalization as valuation
        sim_data.append([avg_volatility, avg_interest_rate, prices[-1], sim_valuations[i]])

    # Convert simulation data to DataFrame
    sim_data_df = pd.DataFrame(sim_data, columns=['Volatility', 'InterestRate', 'FinalPrice', 'Valuation'])

    # Plot the simulated stock price paths
    plt.figure(figsize=(14, 8))
    for i in range(min(num_simulations, 100)):  # Plot a subset of the paths
        plt.plot(range(T), sim_prices[i, :], color='grey', alpha=0.1)
    plt.title(f'Monte Carlo Simulated Stock Price Paths for GVKEY: {target_gvkey}')
    plt.xlabel('Year')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.show()

    # Plot the distribution of simulated valuations
    plt.figure(figsize=(14, 8))
    plt.hist(sim_valuations, bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribution of Simulated Company Valuations for GVKEY: {target_gvkey}')
    plt.xlabel('Valuation')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    mean_valuation = np.mean(sim_valuations)
    std_valuation = np.std(sim_valuations)
    print(f"Mean Valuation for GVKEY {target_gvkey}: ${mean_valuation:.2f}")
    print(f"Standard Deviation of Valuation for GVKEY {target_gvkey}: ${std_valuation:.2f}")

    return sim_valuations, sim_prices, sim_data_df


def analyze_simulation_results(sim_valuations, sim_prices, sim_data_df, num_simulations=10000):
    # Analyze the distribution of valuations
    mean_valuation = np.mean(sim_valuations)
    median_valuation = np.median(sim_valuations)
    std_valuation = np.std(sim_valuations)
    percentile_5 = np.percentile(sim_valuations, 5)
    percentile_95 = np.percentile(sim_valuations, 95)

    print(f"Mean Valuation: ${mean_valuation:.2f}")
    print(f"Median Valuation: ${median_valuation:.2f}")
    print(f"Standard Deviation of Valuation: ${std_valuation:.2f}")
    print(f"5th Percentile Valuation: ${percentile_5:.2f}")
    print(f"95th Percentile Valuation: ${percentile_95:.2f}")

    # Identify scenarios with the highest and lowest valuations
    highest_valuation_idx = np.argmax(sim_valuations)
    lowest_valuation_idx = np.argmin(sim_valuations)

    highest_valuation_scenario = sim_prices[highest_valuation_idx]
    lowest_valuation_scenario = sim_prices[lowest_valuation_idx]

    print(f"Highest Valuation Scenario: {highest_valuation_scenario}")
    print(f"Lowest Valuation Scenario: {lowest_valuation_scenario}")

    return {
        "mean_valuation": mean_valuation,
        "median_valuation": median_valuation,
        "std_valuation": std_valuation,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "highest_valuation_scenario": highest_valuation_scenario,
        "lowest_valuation_scenario": lowest_valuation_scenario,
        "sim_data_df": sim_data_df
    }


def formulate_recommendations(analysis_results):
    mean_valuation = analysis_results["mean_valuation"]
    percentile_5 = analysis_results["percentile_5"]
    percentile_95 = analysis_results["percentile_95"]

    recommendations = []

    # Strategic Recommendation 1: Hedging Strategies
    recommendations.append("Implement hedging strategies to mitigate the risk of unfavorable market movements, especially given the 5th percentile valuation.")

    # Strategic Recommendation 2: Adjustments to Capital Structure
    recommendations.append("Consider adjustments to the capital structure to optimize the cost of capital, leveraging the mean and median valuation insights.")

    # Strategic Recommendation 3: Dynamic Pricing Approaches
    recommendations.append("Adopt dynamic pricing approaches for acquisitions to take advantage of market conditions, informed by the distribution of valuations.")

    # Strategic Recommendation 4: Risk Management
    recommendations.append("Develop a robust risk management framework that considers the volatility and uncertainty highlighted in the 5th and 95th percentile valuations.")

    print("\nStrategic Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")

    return recommendations


def machine_learning_analysis(sim_data_df):
    # Prepare data for machine learning
    X = sim_data_df[['Volatility', 'InterestRate', 'FinalPrice']]
    y = sim_data_df['Valuation']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Feature importance
    feature_importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_importance_df)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from Random Forest Regressor')
    plt.show()

    return model, feature_importance_df


# Plot stock prices around M&A events for Activision Blizzard (gvkey 180405)
plot_stock_prices_around_ma(final_data_with_ma, target_gvkey=180405)

# Generate and plot Brownian Motion paths for Activision Blizzard
generate_brownian_motion(aggregated_data, target_gvkey=180405)

# Plot real vs simulated prices (including GBM) for Activision Blizzard
plot_real_vs_simulated_prices(aggregated_data, target_gvkey=180405)

# Calculate volatility impact for Activision Blizzard (gvkey 180405)
calculate_volatility_impact(final_data_with_ma, target_gvkey=180405)

# Calculate risk and return for Activision Blizzard (gvkey 180405)
calculate_risk_return(final_data_with_ma, target_gvkey='180405')

# Run Monte Carlo valuation
sim_valuations, sim_prices, sim_data_df = monte_carlo_valuation_stock(aggregated_data, target_gvkey=180405)

# Analyze the simulation results
analysis_results = analyze_simulation_results(sim_valuations, sim_prices, sim_data_df)

# Formulate recommendations based on the analysis
recommendations = formulate_recommendations(analysis_results)

# Perform machine learning analysis
model, feature_importance_df = machine_learning_analysis(analysis_results["sim_data_df"])
