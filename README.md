# 📈 Deep Reinforcement Learning for Quantitative Trading 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![FinRL](https://img.shields.io/badge/FinRL-Supported-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A robust, fully-automated Deep Reinforcement Learning (DRL) quantitative trading framework. This project implements the **DDPG**, **SAC**, **PPO** algorithm to dynamically manage a portfolio of Dow Jones 30 (DJIA) constituent stocks, achieving positive Alpha during the 2021-2023 bear market.

## 🌟 Key Features & Technical Highlights

* **Outperforming the Benchmark**: Achieved a positive cumulative return (+1.24%) with a strictly controlled maximum drawdown (-10.62%) during the 2021-2023 market downturn, outperforming the Dow Jones benchmark (-5.37% / -21.94% Max DD).
* **Strict Walk-Forward Validation**: Implements a highly realistic Rolling-Window backtesting mechanism (e.g., Train for 252 days, Test for 20 days) to prevent data leakage and overfitting.
* **Action Penalty Mechanism**: Introduced a custom "Diamond Hands" penalty in the reward function to penalize excessive high-frequency trading, effectively reducing transaction friction costs.
* **Seamless FinRL & ElegantRL Integration**: Developed a robust `ElegantFinRLWrapper` to resolve severe compatibility issues between FinRL's environment observations and ElegantRL's strict `kwargs_filter` parameter interceptions.
* **PyTorch 2.6 Security Patch**: Includes a global patch to seamlessly load `.pth` weights, bypassing the new `weights_only=True` restriction in PyTorch 2.6+.

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
   cd Your-Repo-Name
