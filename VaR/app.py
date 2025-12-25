
IS_STREAMLIT = True

# 1. IMPORTS


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.stats import norm, chi2, t
from arch import arch_model
from pandas_datareader import data as pdr
import datetime
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


# 2. SIDEBAR CONTROLS


st.sidebar.header("⚙ Risk Model Controls")

st.sidebar.markdown("---")
st.sidebar.header("Date Selection")

start_date_input = st.sidebar.date_input(
    "Start Date",
    datetime.date(2018, 1, 1)
)

end_date_input = st.sidebar.date_input(
    "End Date",
    datetime.date.today()
)

confidence = st.sidebar.slider(
    "Confidence Level",
    0.90, 0.99, 0.95, 0.01
)

rolling_window = st.sidebar.selectbox(
    "Rolling VaR Window (days)",
    [125, 250, 500],
    index=1
)

stress_shock = st.sidebar.slider(
    "Stress Shock (%)",
    1, 20, 5
) / 100

mc_sims = st.sidebar.selectbox(
    "Monte Carlo Simulations",
    [5000, 10000, 20000],
    index=1
)

mc_type = st.sidebar.radio(
    "Monte Carlo Type",
    ["Univariate", "Multivariate (Correlated)"]
)

garch_dist = st.sidebar.selectbox(
    "GARCH Distribution",
    ["normal", "t"]
)

st.sidebar.markdown("---")
st.sidebar.header("Model Execution")

run_mc = st.sidebar.checkbox("Run Monte Carlo", value=True)
run_garch = st.sidebar.checkbox("Run GARCH", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Visualizations")

show_dist = st.sidebar.checkbox("Return Distributions")
show_mc_plot = st.sidebar.checkbox("Monte Carlo Plot")
show_garch_plot = st.sidebar.checkbox("GARCH Volatility")
show_backtest_plot = st.sidebar.checkbox("Kupiec Backtest Plot")
show_stress_plot = st.sidebar.checkbox("Stress Test Plot")


# 3. DATE VALIDATION


start_date = datetime.datetime.combine(start_date_input, datetime.time.min)
end_date = datetime.datetime.combine(end_date_input, datetime.time.max)

if start_date >= end_date:
    st.error("Start Date must be before End Date")
    st.stop()


# 4. USER INPUTS


st.title("📊 Portfolio Value at Risk (VaR) Dashboard")

tickers = st.text_input(
    "Tickers (comma-separated)",
    "AAPL,MSFT"
).split(",")

weights_input = st.text_input(
    "Portfolio Weights (comma-separated)",
    "0.5,0.5"
)

try:
    weights = np.array(weights_input.split(","), dtype=float)
    weights = weights / weights.sum()
except Exception:
    st.error("Invalid portfolio weights")
    st.stop()

if len(weights) != len(tickers):
    st.error("Number of weights must match number of tickers")
    st.stop()


# 5. DATA FETCHING


@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    tickers_stooq = [t.strip().upper() + ".US" for t in tickers]
    return pdr.get_data_stooq(
        tickers_stooq,
        start=start,
        end=end
    )["Close"]

try:
    price_data = fetch_data(tickers, start_date, end_date).dropna()
except Exception:
    st.error("Failed to fetch market data")
    st.stop()

returns = price_data.pct_change().dropna()
portfolio_returns = returns @ weights
n_obs = len(portfolio_returns)


# 6. VAR CALCULATIONS

historical_var = np.percentile(
    portfolio_returns,
    (1 - confidence) * 100
)

mean_return = portfolio_returns.mean()
std_return = portfolio_returns.std(ddof=0)

parametric_var = mean_return + norm.ppf(
    1 - confidence
) * std_return

expected_shortfall = (
    portfolio_returns[portfolio_returns < historical_var].mean()
    if (portfolio_returns < historical_var).any()
    else np.nan
)


# 7. MONTE CARLO VAR

monte_carlo_var = np.nan
simulated_returns = None
simulated_portfolio = None

if run_mc:
    if mc_type == "Univariate":
        simulated_returns = np.random.normal(
            mean_return,
            std_return,
            mc_sims
        )
        monte_carlo_var = np.percentile(
            simulated_returns,
            (1 - confidence) * 100
        )
    else:
        mu = returns.mean().values
        cov = returns.cov().values
        L = np.linalg.cholesky(
            cov + 1e-10 * np.eye(len(weights))
        )
        Z = np.random.normal(size=(mc_sims, len(weights)))
        simulated_assets = Z @ L.T + mu
        simulated_portfolio = simulated_assets @ weights
        monte_carlo_var = np.percentile(
            simulated_portfolio,
            (1 - confidence) * 100
        )


# 8. GARCH VAR


garch_var = np.nan
garch_fit = None

@st.cache_resource(show_spinner=False)
def fit_garch(returns, dist):
    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        q=1,
        dist=dist
    )
    return model.fit(disp="off")

if run_garch and n_obs >= 500:
    garch_fit = fit_garch(portfolio_returns, garch_dist)
    forecast = garch_fit.forecast(horizon=1)
    sigma = np.sqrt(forecast.variance.iloc[-1, 0])

    if garch_dist == "t":
        nu = garch_fit.params["nu"]
        garch_var = t.ppf(1 - confidence, nu) * sigma / 100
    else:
        garch_var = norm.ppf(1 - confidence) * sigma / 100


# 9. METRICS


st.markdown("## Risk Metrics Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Historical VaR", f"{historical_var:.2%}")
c2.metric("Parametric VaR", f"{parametric_var:.2%}")
c3.metric("Monte Carlo VaR", f"{monte_carlo_var:.2%}")
c4.metric("GARCH VaR", f"{garch_var:.2%}")

st.metric("Expected Shortfall (CVaR)", f"{expected_shortfall:.2%}")


# 10. ROLLING VAR

rolling_var = portfolio_returns.rolling(
    rolling_window
).quantile(1 - confidence)

fig, ax = plt.subplots()
ax.plot(portfolio_returns, alpha=0.4, label="Daily Portfolio Returns")
ax.plot(rolling_var, color="red", label="Rolling VaR")
ax.set_title("Rolling Portfolio Value at Risk")
ax.set_xlabel("Date")
ax.set_ylabel("Return")
ax.legend()
st.pyplot(fig)


# 11. RETURN DISTRIBUTION 


if show_dist:
    x = np.linspace(
        portfolio_returns.min(),
        portfolio_returns.max(),
        500
    )
    pdf = norm.pdf(x, mean_return, std_return)

    fig, ax = plt.subplots()
    ax.hist(
        portfolio_returns,
        bins=60,
        density=True,
        alpha=0.6,
        label="Empirical Returns"
    )
    ax.plot(x, pdf, label="Normal PDF")
    ax.axvline(
        historical_var,
        color="red",
        linestyle="--",
        label="Historical VaR"
    )
    ax.axvline(
        parametric_var,
        color="black",
        linestyle="--",
        label="Parametric VaR"
    )
    ax.set_title("Portfolio Return Distribution with VaR")
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)


# 12. MONTE CARLO DISTRIBUTION


if show_mc_plot and run_mc:
    fig, ax = plt.subplots()
    ax.hist(
        simulated_returns if mc_type == "Univariate" else simulated_portfolio,
        bins=60,
        alpha=0.7,
        label="Simulated Returns"
    )
    ax.axvline(
        monte_carlo_var,
        color="red",
        linestyle="--",
        label="Monte Carlo VaR"
    )
    ax.set_title("Monte Carlo Simulated Return Distribution")
    ax.set_xlabel("Simulated Portfolio Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)


# 13. GARCH VOLATILITY 


if show_garch_plot and garch_fit is not None:
    fig, ax = plt.subplots()
    ax.plot(
        garch_fit.conditional_volatility,
        label="Conditional Volatility"
    )
    ax.set_title("GARCH(1,1) Conditional Volatility")
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    st.pyplot(fig)


# 14. KUPIEC BACKTEST 

def kupiec_test(returns, var_value, confidence):
    breaches = returns < var_value
    x = breaches.sum()
    n = len(returns)
    p = 1 - confidence

    if x == 0 or x == n:
        return breaches, x, np.nan

    lr = -2 * (
        np.log((1 - p)**(n - x) * p**x) -
        np.log((1 - x/n)**(n - x) * (x/n)**x)
    )
    return breaches, x, 1 - chi2.cdf(lr, df=1)

breach_mask, breaches, p_value = kupiec_test(
    portfolio_returns,
    parametric_var,
    confidence
)

st.subheader("Kupiec Backtesting")

if show_backtest_plot:
    fig, ax = plt.subplots()
    ax.plot(
        portfolio_returns,
        alpha=0.5,
        label="Portfolio Returns"
    )
    ax.scatter(
        portfolio_returns[breach_mask].index,
        portfolio_returns[breach_mask],
        color="red",
        label="VaR Breaches"
    )
    ax.set_title("VaR Backtesting: Kupiec Test")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    st.pyplot(fig)


# 15. BASEL TRAFFIC LIGHT

st.subheader("Basel Traffic Light Test")

if n_obs < 250:
    st.warning("Basel Traffic Light requires 250 trading days")
else:
    if breaches <= 4:
        st.success("🟢 GREEN ZONE — Model Accepted")
    elif breaches <= 9:
        st.warning("🟡 YELLOW ZONE — Capital Multiplier Increase")
    else:
        st.error("🔴 RED ZONE — Model Rejected")


# 16. STRESS TEST 

stress_returns = portfolio_returns - stress_shock

fig, ax = plt.subplots()
ax.plot(
    portfolio_returns,
    label="Original Returns"
)
ax.plot(
    stress_returns,
    label="Stressed Returns"
)
ax.set_title("Stress Testing Scenario")
ax.set_xlabel("Date")
ax.set_ylabel("Return")
ax.legend()
st.pyplot(fig)

# To run the model, In your terminal bash -----> streamlit run app.py   

