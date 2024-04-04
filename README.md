## Project Abstract:
##### In finance, the variability of asset prices are uncertain, and understanding their movements is important for investors and financial institutions to manage their risk. As such, these groups are interested in modeling and forecasting asset price volatility. The baseline, and most often used, model to forecast volatility is the linear Heterogeneous Autoregressive (HAR) model. While the HAR performs well given its simplicity, we suspect that more flexible (i.e. machine learning) models can do better. We seek to improve one day ahead volatility forecasts for the S&P 500 (SPY) using machine learning models: SPY is the largest and most traded Exchange Traded Fund in the US and a bellwether for forecasting stock market volatility.  
##### This work will focus on the Random Forest (RF) framework, while a peer examined Neural Networks (NN). The results of the two ML models will be compared. Additionally, we seek to expand the feature set of the RF model beyond the autoregressive lags found in the HAR. Significant improvement was observed when exogenous inputs containing "market information" were added.  
##### Finally, the one-day-ahead forecasts were used to determine buy/sell signals for an option trading strategy based on next the day's volatility: the RF predictions outperformed a peer's Neural Network, the HAR, and a control over the period of 4/1/19 - 8/10/23, with a volatile but impressive cumulative return of over 3000%.


## Setup

#### To run this project, install it locally using a virtual environment:

bash:
python3 -m venv env
source env/bin/activate # On Windows use `env\Scripts\activate`
pip install -r requirements.txt

#### VSCode Settings

See .vscode/settings.json


## For testing capabilities:
1. 'nano ~/.zshrc'
2. 'export PYTHONPATH={your_path}/VolForecasting_RandomForests'
3. 'source ~/.zshrc'
