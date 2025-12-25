
**📊 Portfolio Value at Risk (VaR) Dashboard**

An interactive **risk analytics dashboard** built using **Streamlit** that computes
portfolio Value at Risk (VaR) using multiple methodologies:

# Portfolio Risk Models
- Historical VaR
- Parametric (Normal) VaR
- Monte Carlo VaR (Univariate & Multivariate)
- GARCH(1,1) VaR(Normal/student-t)
# Risk Echancements
- Expected Shortfall (CVaR)
- Rolling VaR
- Stress Testing
# Model Validation
- Kupiec Backtesting
- Basel Test Classification

The dashboard allows users to dynamically control model parameters
and visualize portfolio risk in real time.

# How to Run the App
In termianl bash --->  *streamlit run app.py*

<!-- ################################################################################################################################ -->

<!--1. Application Mode and Library Imports -->

IS_STREAMLIT = True
# Creating a variable named `IS_STREAMLIT`
# Setting its value to True to indicate the code is running as a Streamlit app
# Using this flag to distinguish Streamlit execution from other environments

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, chi2
from arch import arch_model
from pandas_datareader import data as pdr
import datetime
plt.style.use("seaborn-v0_8")
# Importing the NumPy library for numerical computations, Using NumPy for arrays, mathematical operations, and simulations
# Importing the Pandas library for data manipulation, Using Pandas for handling time series and tabular financial data
# Importing Matplotlib for data visualization, Using Matplotlib to create plots and charts
# Importing the Streamlit library, Using Streamlit to build the interactive web-based dashboard
# Importing the normal distribution and chi-square distribution from SciPy, Using `norm` for VaR and Z-score calculations, Using `chi2` for Kupiec backtesting statistical tests
# Importing the ARCH modeling function, Using `arch_model` to build GARCH volatility models
# Importing the data reader module from pandas_datareader, Using it to fetch historical financial market data
# Importing the datetime module, Using it to define start and end dates for data fetching
# Setting the default plotting style to "seaborn-v0_8", Applying a clean and professional visual theme to all Matplotlib plots



<!--2. SIDEBAR CONTROLS -->

st.sidebar.header("⚙ Risk Model Controls")
# Creating a section header in the Streamlit sidebar
# Labeling the sidebar section as "⚙ Risk Model Controls"
# Grouping all risk-related user inputs under a single header
# Improving readability and organization of the sidebar
# Indicating that the controls below affect the risk models

confidence = st.sidebar.slider(
    "Confidence Level",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01
)
# Creating a variable named `confidence`
# Adding a slider input in the Streamlit sidebar
# Labeling the slider as "Confidence Level" to describe its purpose
# Setting the minimum selectable value to 0.90
# Setting the maximum selectable value to 0.99
# Initializing the slider with a default value of 0.95
# Allowing the slider to move in increments of 0.01
# Storing the value selected by the user in the `confidence` variable
# Using this variable later to control VaR, CVaR, Monte Carlo, and GARCH risk calculations

rolling_window = st.sidebar.selectbox(
    "Rolling VaR Window (days)",
    [125, 250, 500],
    index=1
)
# Creating a variable named `rolling_window`
# Adding a dropdown (selectbox) input in the Streamlit sidebar
# Labeling the dropdown as "Rolling VaR Window (days)"
# Providing a list of selectable options: 125, 250, and 500 days
# Setting the default selected option using `index=1`
# Selecting the value 250 days by default (since index starts from 0)
# Storing the user-selected window size in the `rolling_window` variable
# Using this value later to calculate Rolling VaR over the selected time window

stress_shock = st.sidebar.slider(
    "Stress Shock (%)",
    min_value=1,
    max_value=20,
    value=5
) / 100
# Creating a variable named `stress_shock`
# Adding a slider input in the Streamlit sidebar
# Labeling the slider as "Stress Shock (%)" to indicate percentage input
# Setting the minimum selectable value to 1
# Setting the maximum selectable value to 20
# Setting the default slider value to 5
# Allowing the user to choose a stress shock between 1% and 20%
# Dividing the selected value by 100 to convert percentage into decimal form
# Storing the final decimal shock value in the `stress_shock` variable
# Using this variable later for portfolio stress testing calculations

mc_sims = st.sidebar.selectbox(
    "Monte Carlo Simulations",
    [5000, 10000, 20000, 50000],
    index=1
)
# Creating a variable named `mc_sims`
# Adding a dropdown (selectbox) input in the Streamlit sidebar
# Labeling the dropdown as "Monte Carlo Simulations"
# Providing a list of selectable simulation counts: 5000, 10000, 20000, and 50000
# Setting the default selected option using `index=1`
# Selecting 10000 simulations by default (since index starts from 0)
# Storing the user-selected number of simulations in the `mc_sims` variable
# Using this value later to control the number of Monte Carlo simulations
# Higher values increase accuracy while increasing computational cost

mc_type = st.sidebar.radio(
    "Monte Carlo Type",
    ["Univariate", "Multivariate (Correlated)"]
)
# Creating a variable named `mc_type`
# Adding a radio button input in the Streamlit sidebar
# Labeling the radio buttons as "Monte Carlo Type"
# Providing two selectable options: Univariate and Multivariate (Correlated)
# Allowing the user to choose one Monte Carlo simulation approach
# Storing the selected option as a string in the `mc_type` variable
# Using "Univariate" to simulate portfolio returns directly
# Using "Multivariate (Correlated)" to simulate correlated asset returns
# Controlling which Monte Carlo logic is executed later in the code

garch_dist = st.sidebar.selectbox(
    "GARCH Distribution",
    ["normal", "t"]
)
# Creating a variable named `garch_dist`
# Adding a dropdown (selectbox) input in the Streamlit sidebar
# Labeling the dropdown as "GARCH Distribution"
# Providing two selectable distribution options: normal and t
# Allowing the user to choose the distribution for GARCH model residuals
# Storing the selected distribution as a string in the `garch_dist` variable
# Using "normal" for Gaussian residuals
# Using "t" to model fat-tailed return distributions
# Applying this choice later when fitting the GARCH volatility model



<!--3. USER INPUTS -->

st.title("Portfolio Value at Risk (VaR) Dashboard")
# Displaying the main title of the Streamlit application
# Setting the title text to "Portfolio Value at Risk (VaR) Dashboard"


tickers = st.text_input(
    "Tickers (comma-separated)",
    "AAPL,MSFT"
).split(",")
# Creating a text input field for stock tickers
# Labeling the input as "Tickers (comma-separated)"
# Providing a default value of "AAPL,MSFT"
# Splitting the user input by commas to create a list of tickers
# Storing the list of tickers in the `tickers` variable


weights_input = st.text_input(
    "Portfolio Weights (comma-separated)",
    "0.5,0.5"
)
# Creating a text input field for portfolio weights
# Labeling the input as "Portfolio Weights (comma-separated)"
# Providing a default value of "0.5,0.5"
# Storing the raw string input in the `weights_input` variable


weights = np.array(weights_input.split(","), dtype=float)
weights = weights / weights.sum()
# Converting the weight strings into a NumPy array of floats
# Splitting the input by commas before conversion
# Storing the numerical weights in the `weights` array
# Normalizing the weights so that their sum equals 1
# Ensuring the portfolio allocation is valid


if len(weights) != len(tickers):
    st.error("Number of weights must match number of tickers")
    st.stop()
# Checking whether the number of weights matches the number of tickers
# Displaying an error message if the lengths do not match
# Stopping the execution of the app to prevent calculation errors

start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2024, 1, 1)
# Defining the start date for historical data fetching
# Setting the start date to January 1, 2020
# Defining the end date for historical data fetching
# Setting the end date to January 1, 2024



<!--4. DATA FETCHING & Returun calculation-->

tickers_stooq = [t.strip().upper() + ".US" for t in tickers]
# Creating a new list named `tickers_stooq`
# Stripping extra spaces from each ticker symbol
# Converting each ticker to uppercase
# Appending ".US" to match the Stooq data source format


price_data = pdr.get_data_stooq(
    tickers_stooq,
    start=start_date,
    end=end_date
)["Close"].dropna()
# Fetching historical market data from the Stooq data provider
# Passing the formatted ticker symbols to the data reader
# Specifying the start date for data retrieval
# Specifying the end date for data retrieval
# Selecting only the "Close" price column from the fetched data
# Removing rows with missing values using dropna()
# Storing the cleaned price data in the `price_data` variable

returns = price_data.pct_change().dropna()
# Calculating daily percentage returns from closing prices
# Using pct_change() to compute day-over-day returns
# Removing the first NaN row created by the percentage change using dropna()
# Storing the return series in the `returns` variable

portfolio_returns = returns @ weights
# Computing portfolio-level returns using matrix multiplication @ 
# using @ computes a weighted avg  sum of asset returns by multiplying asset returns by their respective portfolio weights
# Mathematically, it performs:
# Portfolio Return = (Asset1 Return × Weight1) + (Asset2 Return × Weight2) + ...
# Aggregating individual asset returns into a single portfolio return series
# Storing the final portfolio returns in the `portfolio_returns` variable



<!--5. HISTORICAL VaR MODEL -->

historical_var = np.percentile(
    portfolio_returns,
    (1 - confidence) * 100
)
# Creating a variable named `historical_var`
# Using NumPy’s `percentile` function to compute a percentile from historical data
# Passing `portfolio_returns` as the input return distribution
# Calculating the percentile level using (1 - confidence) × 100
# Converting the confidence level into a lower-tail percentile
# For example, a 95% confidence level corresponds to the 5th percentile
# Identifying the worst loss that occurs within the selected confidence level
# Making no distributional assumptions about returns
# Storing the computed Historical VaR value in the `historical_var` variable
# Using this value later for risk reporting and comparison with other VaR models



<!--6. PARAMETRIC (NORMAL) VaR -->

mean_return = portfolio_returns.mean()
# Calculating the average (mean) of the portfolio return series
# Using the mean as the expected daily return
# Storing the result in the `mean_return` variable

std_return = portfolio_returns.std()
# Calculating the standard deviation of the portfolio return series
# Measuring the volatility of portfolio returns
# Storing the result in the `std_return` variable

parametric_var = mean_return + norm.ppf(
    1 - confidence
) * std_return
# Creating a variable named `parametric_var`
# Using a parametric approach assuming returns follow a normal distribution
# Calling `norm.ppf` to obtain the Z-score for the lower tail of the distribution
# Passing (1 - confidence) to target the loss quantile
# Scaling the Z-score by the portfolio standard deviation
# Adding the scaled volatility to the mean return
# Computing the portfolio Value at Risk under normality assumption
# Storing the final Parametric VaR value in the `parametric_var` variable



<!--7. EXPECTED SHORTFALL (CVaR) -->

expected_shortfall = portfolio_returns[
    portfolio_returns < historical_var
].mean()
# Creating a variable named `expected_shortfall`
# Selecting portfolio returns that are worse than the Historical VaR threshold
# Filtering the return series to include only tail-loss observations
# Computing the mean of these extreme losses
# Measuring the average loss given that the VaR threshold has been breached
# Capturing tail risk beyond the VaR level
# Using Historical VaR as the cutoff point for the calculation
# Storing the resulting Expected Shortfall value in the `expected_shortfall` variable
# Providing a more conservative risk measure than VaR



<!--8. MONTE CARLO VaR --> -->

if mc_type == "Univariate":
# Checking which Monte Carlo method the user selected from the sidebar
# Comparing the value of `mc_type` with the string "Univariate"

    simulated_returns = np.random.normal(
        mean_return,
        std_return,
        mc_sims
    )
# If the Univariate Monte Carlo option is selected:
# Simulating portfolio returns directly using a normal distribution
# Using the portfolio mean return as the distribution mean
# Using the portfolio standard deviation as the distribution volatility
# Generating a number of simulated returns equal to `mc_sims`
# Storing the simulated returns in the `simulated_returns` variable
   
    monte_carlo_var = np.percentile(
        simulated_returns,
        (1 - confidence) * 100
    )
# Computing the Value at Risk from the simulated return distribution
# Using NumPy’s percentile function to extract the lower-tail loss
# Calculating the percentile using (1 - confidence) × 100
# Storing the Monte Carlo VaR result in the `monte_carlo_var` variable

else:
    cov = returns.cov()
    L = np.linalg.cholesky(cov)
# If the Multivariate (Correlated) Monte Carlo option is selected:
# Calculating the covariance matrix of asset returns
# Capturing the dependency structure between assets
# Applying Cholesky decomposition to the covariance matrix
# Transforming independent random variables into correlated variables

    Z = np.random.normal(size=(mc_sims, len(tickers)))
    simulated_assets = Z @ L.T
    simulated_portfolio = simulated_assets @ weights
# Generating a matrix of independent standard normal random variables
# Setting the matrix dimensions to (number of simulations × number of assets)
# Creating correlated asset return simulations using matrix multiplication
# Preserving historical correlations between assets
# Aggregating simulated asset returns into portfolio-level returns
# Applying portfolio weights to each simulated asset return

    monte_carlo_var = np.percentile(
        simulated_portfolio,
        (1 - confidence) * 100
    )
# Computing the Value at Risk from the simulated portfolio returns
# Extracting the lower-tail percentile based on the confidence level
# Storing the final Monte Carlo VaR value in the `monte_carlo_var` variable



<!--9. GARCH(1,1) VaR MODEL -->

garch_model = arch_model(
    portfolio_returns * 100,
    vol="Garch",
    p=1,
    q=1,
    dist=garch_dist
)
# Creating a GARCH(1,1) model using the `arch_model` function
# Passing portfolio returns multiplied by 100 to scale returns to percentages
# Specifying the volatility model type as "Garch"
# Setting p = 1 to include one lag of past squared returns (ARCH term)
# Setting q = 1 to include one lag of past conditional variance (GARCH term)
# Selecting the residual distribution based on user input (`garch_dist`)
# Storing the GARCH model specification in the `garch_model` variable

garch_fit = garch_model.fit(disp="off")
# Fitting the GARCH model to the portfolio return data
# Estimating model parameters using maximum likelihood
# Suppressing convergence output by setting disp="off"
# Storing the fitted model in the `garch_fit` variable

forecast = garch_fit.forecast(horizon=1)
# Forecasting future volatility using the fitted GARCH model
# Setting the forecast horizon to 1 period (next trading day)
# Extracting the forecasted variance value from the results

garch_sigma = np.sqrt(forecast.variance.iloc[-1, 0])
# Taking the square root of the forecasted variance to obtain volatility
# Storing the forecasted volatility in the `garch_sigma` variable

garch_var = norm.ppf(1 - confidence) * garch_sigma / 100
# Computing the GARCH-based Value at Risk
# Using the normal distribution quantile for the selected confidence level
# Scaling the volatility by the Z-score
# Dividing by 100 to convert percentage volatility back to return units
# Storing the final GARCH VaR value in the `garch_var` variable



<!--10. RISK METRICS DISPLAY -->

st.subheader("Risk Metrics")
# Creating a subheader in the Streamlit app titled "Risk Metrics"
# Displaying this section title above the risk values

c1, c2, c3, c4 = st.columns(4)
# Creating four columns in the Streamlit layout
# Assigning each column to variables c1, c2, c3, and c4

c1.metric("Historical VaR", f"{historical_var:.2%}")
# Displaying the Historical VaR value in the first column
# Formatting the Historical VaR as a percentage with two decimal places

c2.metric("Parametric VaR", f"{parametric_var:.2%}")
# Displaying the Parametric VaR value in the second column
# Formatting the Parametric VaR as a percentage with two decimal places

c3.metric("Monte Carlo VaR", f"{monte_carlo_var:.2%}")
# Displaying the Monte Carlo VaR value in the third column
# Formatting the Monte Carlo VaR as a percentage with two decimal places

c4.metric("GARCH VaR", f"{garch_var:.2%}")
# Displaying the GARCH VaR value in the fourth column
# Formatting the GARCH VaR as a percentage with two decimal places

st.metric("Expected Shortfall (CVaR)", f"{expected_shortfall:.2%}")
# Displaying the Expected Shortfall (CVaR) below the columns
# Formatting the CVaR value as a percentage with two decimal places
# Presenting CVaR separately to emphasize tail risk



<!--11. ROLLING VaR -->

rolling_var = portfolio_returns.rolling(
    rolling_window
).quantile(1 - confidence)
# Creating a rolling window object on the portfolio return series
# Using the window size selected by the user (`rolling_window`)
# Computing the rolling quantile at the (1 - confidence) level
# Calculating a time-varying Historical VaR
# Storing the rolling VaR series in the `rolling_var` variable

fig, ax = plt.subplots()
ax.plot(portfolio_returns, alpha=0.4, label="Portfolio Returns")
ax.plot(rolling_var, color="red", label="Rolling VaR")
ax.legend()
ax.set_title("Rolling VaR")
# Creating a Matplotlib figure and axis for plotting
# Initializing the plot canvas
# Plotting the portfolio return time series
# Setting transparency using alpha=0.4 for better visual clarity
# Labeling the line as "Portfolio Returns"
# Plotting the rolling VaR time series
# Coloring the rolling VaR line red for emphasis
# Labeling the line as "Rolling VaR"
# Adding a legend to distinguish plotted lines
# Setting the plot title to "Rolling VaR"

st.pyplot(fig)
# Rendering the Matplotlib figure inside the Streamlit application
# Displaying the rolling VaR visualization to the user



<!--12. KUPIEC BACKTEST -->

def kupiec_test(returns, var_value, confidence):
# Defining a function named `kupiec_test`
# Accepting three inputs: returns, VaR value, and confidence level

    breaches = returns < var_value
# Creating a boolean series identifying VaR breaches
# Marking True where returns are less than the VaR threshold

    x = breaches.sum()
    n = len(returns)
    p = 1 - confidence
# Counting the total number of VaR breaches
# Storing the number of breaches in the variable `x`
# Calculating the total number of observations
# Storing the length of the return series in the variable `n`

    eps = 1e-10
    x = np.clip(x, eps, n - eps)
# Calculating the expected breach probability
# Using (1 - confidence) as the theoretical failure rate
# Defining a small epsilon value to avoid numerical errors
# Preventing division by zero and log(0) issues

    lr = -2 * (
        np.log((1 - p)**(n - x) * p**x) -
        np.log((1 - x/n)**(n - x) * (x/n)**x)
    )
# Clipping the breach count to a safe numerical range
# Ensuring stable likelihood calculations
# Computing the likelihood ratio statistic for the Kupiec test
# Comparing observed breach frequency with expected frequency
# Applying the log-likelihood formulation of the test

    p_value = 1 - chi2.cdf(lr, df=1)
    return int(breaches.sum()), p_value
    breaches, p_value = kupiec_test(
    portfolio_returns,
    parametric_var,
    confidence
    )
# Computing the p-value using the chi-square distribution
# Using 1 degree of freedom for the Kupiec test
# Measuring whether the VaR model is statistically valid
# Returning the total number of breaches and the p-value
# Converting breach count to an integer for clean display
# Running the Kupiec test using portfolio returns
# Passing the parametric VaR value as the VaR threshold
# Passing the selected confidence level

st.subheader("Kupiec Backtesting")
st.write(f"Number of breaches: {breaches}")
st.write(f"P-value: {p_value:.4f}")
# Displaying a subheader titled "Kupiec Backtesting"
# Writing the number of observed VaR breaches to the app
# Writing the Kupiec test p-value formatted to four decimals



<!--13. BASEL TRAFFIC LIGHT -->

st.subheader("Basel Test")
# Creating a subheader in the Streamlit app titled "Basel Test"
# Displaying the regulatory backtesting classification section

if breaches <= 4:
    st.success("GREEN ZONE")
elif breaches <= 9:
    st.warning("YELLOW ZONE")
else:
    st.error("RED ZONE")
# Checking if the number of VaR breaches is less than or equal to 4
# Classifying the model as GREEN ZONE
# Indicating acceptable model performance using a success message

# Checking if the number of VaR breaches is between 5 and 9
# Classifying the model as YELLOW ZONE
# Indicating caution using a warning message

# Checking if the number of VaR breaches exceeds 9
# Classifying the model as RED ZONE
# Indicating poor model performance using an error message



<!--14. STRESS TESTING -->

stress_returns = portfolio_returns - stress_shock
# Creating a new return series for stress testing
# Applying a negative stress shock to portfolio returns
# Simulating adverse market conditions

stress_var = np.percentile(
    stress_returns,
    (1 - confidence) * 100
)
# Calculating the stressed Value at Risk
# Using NumPy’s percentile function on stressed returns
# Computing the lower-tail percentile based on the confidence level
# Storing the stressed VaR value in the `stress_var` variable

st.subheader("Stress Testing")
# Creating a subheader titled "Stress Testing"
# Displaying the stress testing results section

st.metric(
    f"Stress VaR ({int(stress_shock*100)}% shock)",
    f"{stress_var:.2%}"
)
# Displaying the Stress VaR metric in the Streamlit app
# Including the shock percentage in the metric label
# Formatting the stressed VaR value as a percentage with two decimals

