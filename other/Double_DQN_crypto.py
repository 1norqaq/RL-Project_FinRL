import os
import pandas as pd
import numpy as np
import torch
import gymnasium as gym
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 安全补丁 (适配较新版本 PyTorch 的加载要求)
# ---------------------------------------------------------
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# ---------------------------------------------------------
# 2. 导入 ElegantRL 组件
# ---------------------------------------------------------
from elegantrl.agents import AgentDoubleDQN
from elegantrl.train.run import train_agent
try:
    from elegantrl.train.config import Config as Arguments
except ImportError:
    from elegantrl.train.config import Arguments

# ---------------------------------------------------------
# 3. 核心：NumPy 极速版离散高频环境 (解决采样瓶颈)
# ---------------------------------------------------------
class FastDiscreteCryptoEnv(gym.Env):
    def __init__(self, df, indicators, initial_amount=1000000, hmax=5, cost_pct=0.0002, env_name="FastCryptoEnv", **kwargs):
        super().__init__()
        self.env_name = env_name
        
        # 将 DataFrame 提前转换为 NumPy 矩阵，消除 CPU 索引瓶颈！
        self.prices = df['close'].values.astype(np.float32)
        self.features = df[indicators].values.astype(np.float32)
        self.max_step = len(df) - 1
        
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.cost_pct = cost_pct
        
        # 动作空间：0(卖出), 1(持有), 2(买入)
        self.action_space = gym.spaces.Discrete(3)
        self.action_dim = 3
        
        # 状态空间：[账户余额, 当前价格, 持有份额] + [特征指标...]
        self.state_dim = 3 + len(indicators)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
        
        # 记录内部状态
        self.day = 0
        self.balance = self.initial_amount
        self.shares = 0
        self.asset_memory = []
        
    def reset(self, seed=None, options=None):
        self.day = 0
        self.balance = self.initial_amount
        self.shares = 0
        self.asset_memory = [self.initial_amount]
        return self._get_obs(), {}
        
    def _get_obs(self):
        # 纯 NumPy 拼接，速度极快
        obs = np.concatenate(([self.balance, self.prices[self.day], self.shares], self.features[self.day]))
        return obs.astype(np.float32)
        
    def step(self, action):
        self.day += 1
        if self.day >= self.max_step:
            return self._get_obs(), 0.0, True, False, {}
            
        price = self.prices[self.day]
        trade_shares = 0
        
        # 动作执行逻辑
        if action == 0 and self.shares > 0: # 卖出
            trade_shares = -min(self.shares, self.hmax)
        elif action == 2: # 买入
            # 👇 防除零护盾：如果碰到异常脏数据(价格为0)，直接放弃买入，防止引发 OverflowError
            if price > 0:
                max_buy = int(self.balance / (price * (1 + self.cost_pct)))
                trade_shares = min(max_buy, self.hmax)
            else:
                trade_shares = 0
            
        # 计算交易成本与余额更新
        trade_amount = trade_shares * price
        cost = abs(trade_amount) * self.cost_pct
        self.balance -= (trade_amount + cost)
        self.shares += trade_shares
        
        # 计算奖励 (资产净值变化)
        current_asset = self.balance + self.shares * price
        prev_asset = self.asset_memory[-1]
        reward = current_asset - prev_asset
        self.asset_memory.append(current_asset)
        
        # Reward 缩放，帮助神经网络收敛
        reward = reward * 1e-4
        done = self.day >= self.max_step
        
        return self._get_obs(), float(reward), done, False, {}

# ---------------------------------------------------------
# 4. 数据处理逻辑
# ---------------------------------------------------------
def load_local_crypto_data(csv_path="./data/BTC_1sec.csv", npy_path="./data/BTC_1sec_predict.npy"):
    print(f"[INFO] 正在加载高频数据...")
    if not os.path.exists(csv_path) or not os.path.exists(npy_path):
        print("[WARNING] 文件丢失，生成 MOCK 数据用于测试。请确保 /data 目录下存在所需文件。")
        np.random.seed(42)
        mock_len = 5000
        raw_df = pd.DataFrame({'SystemTime': pd.date_range('2021-04-07', periods=mock_len, freq='S'),
                               'MidPrice': np.cumsum(np.random.randn(mock_len)) + 50000})
        features = np.random.randn(mock_len, 8) 
    else:
        raw_df = pd.read_csv(csv_path)
        features = np.load(npy_path)

    # 对齐长度（如果有 RNN Lookback Window）
    diff = len(raw_df) - len(features)
    if diff > 0:
        raw_df = raw_df.iloc[diff:].reset_index(drop=True)

    finrl_df = pd.DataFrame()
    price_col = 'MidPrice' if 'MidPrice' in raw_df.columns else raw_df.columns[1]
    finrl_df['close'] = pd.to_numeric(raw_df[price_col], errors='coerce').astype(float)
    
    feature_cols = [f"rnn_feature_{i}" for i in range(features.shape[1])]
    for i, col in enumerate(feature_cols):
        finrl_df[col] = features[:, i].astype(float)

    # 填充缺失值，防止脏数据影响特征输入
    finrl_df = finrl_df.fillna(method='ffill').fillna(0.0)
    return finrl_df, feature_cols

def split_train_test_physical(df):
    test_size = 2700  # 论文标准：测试集为最后 45 分钟的数据
    df_train = df.iloc[:-test_size].copy()
    df_test = df.iloc[-test_size:].copy()
    return df_train, df_test

# ---------------------------------------------------------
# 5. DDQN 训练与评估配置
# ---------------------------------------------------------
def setup_ddqn_args(env_params, cwd_path):
    args = Arguments(agent_class=AgentDoubleDQN, env_class=FastDiscreteCryptoEnv)
    args.env_args = env_params
    args.env_name = env_params['env_name']

    # [cite_start]论文设定：三个 128 单元的前馈神经网络隐藏层 [cite: 771]
    args.net_dims = (128, 128, 128)
    
    args.state_dim = env_params['state_dim']
    args.action_dim = env_params['action_dim']
    args.if_discrete = True

    # [cite_start]论文设定：学习率极为保守，设定为 2e-6 [cite: 772]
    args.learning_rate = 2e-6
    args.batch_size = 512

    args.target_step = 2000
    # 极速环境下，跑 100000 步通常只要几分钟
    args.break_step = 100000  
    args.worker_num = 1
    args.eval_proc_num = 0    # 避免 Windows 平台下的评估器多进程闪退
    args.cwd = cwd_path
    args.if_remove = True
    return args

def real_test_inference(test_df, indicators, args, env_params):
    # 构建纯 NumPy 推理环境
    env = FastDiscreteCryptoEnv(
        df=test_df, 
        indicators=indicators, 
        cost_pct=env_params['cost_pct']
    )
    
    agent = AgentDoubleDQN(args.net_dims, args.state_dim, args.action_dim)
    
    # 强制加载最新的模型权重
    model_path_act = f"{args.cwd}/act.pth"
    model_path_actor = f"{args.cwd}/actor.pth"
    
    if os.path.exists(model_path_act):
        agent.act.load_state_dict(torch.load(model_path_act, map_location=agent.device))
        print(f"[INFO] 成功加载模型权重: {model_path_act}")
    elif os.path.exists(model_path_actor):
        agent.act.load_state_dict(torch.load(model_path_actor, map_location=agent.device))
        print(f"[INFO] 成功加载模型权重: {model_path_actor}")
    else:
        print(f"[WARNING] 未找到训练权重! 模型将盲目行动。")
        
    agent.act.eval()
    agent.explore_rate = 0.0 # 考场模式：不使用任何 epsilon-greedy 探索

    res = env.reset()
    state = res[0] if isinstance(res, tuple) else res
    done = False

    print("[INFO] 正在执行 45 分钟逐秒回测推理...")
    while not done:
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action = agent.act(s_tensor).argmax(dim=0).detach().cpu().item()

        step_res = env.step(action)
        state, reward, done, trunc, _ = step_res if len(step_res)==5 else (*step_res[:3], False, {})
        done = done or trunc

    # 导出测试结果
    result_df = pd.DataFrame({'account_value': env.asset_memory})
    return result_df

# ---------------------------------------------------------
# 6. 主程序入口
# ---------------------------------------------------------
if __name__ == '__main__':
    # 1. 加载并拆分数据
    df, PAPER_INDICATORS = load_local_crypto_data()
    df_train, df_test = split_train_test_physical(df)

    # 2. 计算出框架需要的必备参数
    state_dim = 3 + len(PAPER_INDICATORS)
    action_dim = 3
    max_step = len(df_train) - 1

    # 3. 补齐 ElegantRL 多进程装配环境所必需的参数
    env_params = {
        "env_name": "FastCryptoEnv",
        "df": df_train,
        "indicators": PAPER_INDICATORS,
        "cost_pct": 0.0002,           # 降低测试门槛，防止“永远持仓不交易”的策略坍塌
        "state_dim": state_dim,       # 解决 KeyError 的关键
        "action_dim": action_dim,     
        "if_discrete": True,          
        "max_step": max_step,         
        "target_return": 10.0         
    }

    print(f"\n[STEP 1] 开始 DDQN 极速训练...")
    cwd_path = "./checkpoints/ddqn_crypto_hft_fast"
    os.makedirs(cwd_path, exist_ok=True)
    args = setup_ddqn_args(env_params, cwd_path)

    try:
        # 启动训练
        train_agent(args)
        print("\n[✓] 训练已彻底完成！")
    except Exception as e:
        import traceback
        print(f"\n[WARNING] 训练阶段遇到异常，堆栈信息如下:")
        traceback.print_exc()

    print(f"\n[STEP 2] 准备生成回测结果...")
    
    try:
        # 执行测试并提取真正的资金曲线结果
        test_results = real_test_inference(df_test, PAPER_INDICATORS, args, env_params)
        
        output_file = "ddqn_crypto_hft_results.csv"
        test_results.to_csv(output_file, index=False)
        print(f"\n[✓] 成功！结果已保存至 '{output_file}'。")
        print(f"======================================")
        print(f"最终资金曲线前5条预览:\n{test_results.head()}")
        print(f"最终资金曲线最后5条预览:\n{test_results.tail()}")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 回测推理阶段崩溃:")
        traceback.print_exc()
