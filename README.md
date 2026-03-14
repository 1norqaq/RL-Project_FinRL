# 📈 Deep Reinforcement Learning for Quantitative Trading 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![FinRL](https://img.shields.io/badge/FinRL-Supported-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A robust, fully-automated Deep Reinforcement Learning (DRL) quantitative trading framework. This project implements the **DDPG**, **SAC**, **PPO** algorithm to dynamically manage a portfolio of Dow Jones 30 (DJIA) constituent stocks, achieving positive Alpha during the 2021-2023 bear market.


## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/1norqaq/RL-Project_FinRL.git](https://github.com/1norqaq/RL-Project_FinRL.git)
   cd RL-Project_FinRL
   ```

2. Environment setup:
   Follow the instructions from Github - [FinRL](https://github.com/AI4Finance-Foundation/FinRL) and [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL).

## 🚀 Usage

### Task 1: Fixed Window Stock Trading

Following FinRL Contest 2023 rules, agents are trained on OHLCV data for
30 DJI stocks from 01/07/2010 to 10/24/2023 (3352 days). And test on two different peri-
ods, including 10/25/2023-11/14/2023 (15 trading days) and 11/15/2023-11/12-2023 (6 trading
days). The training and testing process is implemented in `stock_tot.ipynb`, which contains DDPG, SAC, PPO, Ensemble algorithms and the DJIA (benchmark).



### Task 2: Rolling Window Stock Trading

To systematically address market non-stationarity over long
horizons, we implement a Walk-Forward Rolling Window framework. Each window comprises three distinct phases: a Training phase (252 days) for initial
policy optimization, a Validation phase (20 days) for performance evaluation and agent selec-
tion, and a Testing phase (20 days) for simulated real-world execution. All the algorithms are implemented in `rolling_window/` directory.

## 📂 Project Structure

```RL-Project_FinRL/
├── README.md
├── stock_tot.ipynb
├── rolling_window/
   |-- sac_rolling_final.py: Implements the SAC algorithm for rolling window stock trading.
   |-- ddpg_rolling_final.py: Implements the DDPG algorithm for rolling window stock trading.
   |-- ppo_rolling_final.py: Implements the PPO algorithm for rolling window stock trading.
   |-- ensemble_run.py: Contains the main function to run the ensemble strategy.
   |-- ensemble_plot.py: Contains functions for plotting the results of the ensemble strategy.
```