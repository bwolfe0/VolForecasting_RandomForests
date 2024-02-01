## This project has a few goals:
1. Use random forest regression to improve OOS (MSE, MAE) performance relative to linear HAR model for Realized Variance (RV) forecasting.
2. Deploy the RV predictions from different models and use them to execute a ODTE option trading strategy based on estimated next day volatility.
3. Expand the random forest model's feature set with separate focuses on interpretability and result imporvement.


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