"""
倒立摆（CartPole-v0）强化学习入门示例
----------------------------------------------------
本文件演示强化学习最基础的几个环节：
1. 环境交互：创建环境、env.reset() 得到初始状态、env.step() 执行一步动作
2. 策略网络 PolicyNet：输入 4 维状态，输出 2 个动作的概率分布 π_θ(a|s)
3. 智能体 Agent：按策略网络采样动作，在环境中完整跑一个回合（rollout 采样轨迹）
4. 回报计算：逆序累加带折扣因子 γ 的轨迹回报 G_0，为后续 REINFORCE 算法做准备
注意：本文件中的策略网络是随机初始化的（未训练），rollout 得到的只是随机策略的轨迹
"""

# 导入 OpenAI Gym 库，提供强化学习环境
import gym
# torch：张量计算与自动求导框架，策略网络基于它构建
import torch
# torch.nn：神经网络模块与层（nn.Module / nn.Linear）
import torch.nn as nn
# torch.nn.functional：函数式接口（relu、softmax 等激活函数）
import torch.nn.functional as F

# 创建一个倒立摆环境：CartPole 是小车-杆系统，目标是通过左右移动小车保持杆子竖直
# v1 版本规则：杆子倾斜超过 12°、小车滑出轨道边界，或单回合达到 500 步时回合结束
env = gym.make('CartPole-v0')

# 将环境重置为初始状态 S0
# 注意版本差异：gym >= 0.26 / gymnasium 中 reset() 返回 (obs, info) 元组；旧版 gym 直接返回 ndarray
state = env.reset()

# 打印初始状态：CartPole 的状态是 4 维向量 [小车位置, 小车速度, 杆角度, 杆角速度]
print(state)

action = 0  # 向左推（CartPole 动作空间只有 2 个离散动作：0=向左推，1=向右推）
# env.step(action) 在环境中执行一步动作，返回一个 4 元组
# next_state: 执行动作后的下一个状态 S_{t+1}
# reward: 即时奖励 R_t（CartPole 每存活一步 +1）
# done: 本回合是否结束（杆倒 / 小车越界 / 达到 500 步上限）
# _: 调试信息（新版 gym 中是 truncated 等额外信息，这里用不到）
next_state, reward, done, _ = env.step(action)

print(f"状态S1:{next_state}")  # 观察执行动作后的新状态
print(f"即时奖励R0:{reward}")  # 观察这一步获得的即时奖励
print(f"是否结束:{done}")  # 第一步动作通常不会导致回合结束，应为 False


class PolicyNet(nn.Module):
    """
    策略网络类：输入状态，输出动作的概率分布 π_θ(A_t|S_t)
    网络结构：4 -> 128 -> 2 的两层 MLP，最后一层 softmax 将 logits 归一化为概率分布
    """

    def __init__(self):
        super().__init__()
        # 输入状态：CartPole 的 4 维状态向量 [小车位置, 小车速度, 杆角度, 杆角速度]
        self.l1 = nn.Linear(4, 128)  # 第 1 层全连接：4 维状态 -> 128 维隐藏特征
        # 输出层：128 维隐藏特征 -> 2 维 logits，对应 2 个动作
        self.l2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.l1(x)  # shape: (B, 4) -> (B, 128)
        x = F.relu(x)  # shape: (B, 128)，relu 激活函数引入非线性
        x = self.l2(x)  # shape: (B, 128) -> (B, 2)，得到两个动作的 logits
        x = F.softmax(x, dim=-1)  # shape: (B, 2)，沿最后一维做 softmax，把 logits 变成和为 1 的概率分布
        return x


# ================ 下面单独测试 PolicyNet 的前向输出 ================
state = env.reset()  # 重新开始一个新回合，得到初始状态 S0
policy = PolicyNet()  # 实例化一个策略网络（此时参数为随机初始值）
# 手动前向传播一次，验证网络输出是否正确：
# torch.tensor(state)：numpy 状态向量 -> torch 张量，形状 (4,)
# unsqueeze(0)：在第 0 维前插入 batch 维，(4,) -> (1, 4)，因为网络输入要求带 batch 维
# squeeze(0)：去掉 batch 维，(1, 2) -> (2,)
probs = policy(
    torch.tensor(state).unsqueeze(0)
).squeeze(0)
print(probs)  # 形状 (2,)，两个动作的概率分布 [P(左推), P(右推)]，随机初始化下各约 0.5


class Agent:
    """
    智能体类：包含策略网络和选择动作的方法
    目前策略网络未训练（随机初始化），rollout 采到的是随机策略的轨迹；
    后续可在此类上实现 REINFORCE 等策略梯度算法来训练 self.pi
    """

    def __init__(self):
        self.gamma = 0.98  # 折扣因子 gamma：越远的奖励权重越低，γ 越接近 1 越看重长期回报
        self.pi = PolicyNet()  # 策略网络 π_θ
        self.lr_pi = 0.001  # 策略网络的学习率
        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=self.lr_pi)  # Adam 优化器（本文件尚未训练，为后续更新参数做准备）

    def get_action(self, state):
        """
            环境state：                  (4,)        → [s0,s1,s2,s3]
            unsqueeze(0)                (1,4)       → [[s0,s1,s2,s3]]   # 适配网络batch输入
            self.pi(state) 网络推理      (1,2)       → [[0.6,0.4]]
            squeeze(0)                  (2,)        → [0.6,0.4]         # 拆掉batch维，用于采样
        """
        # 将 numpy 状态从形状 (4,) 转为 torch 张量并添加 batch 维，变成 (1, 4)
        state = torch.tensor(state).unsqueeze(0)
        # 前向传播输出动作概率分布 π_θ(a|s)，形状 (1, 2)，再 squeeze 去掉 batch 维得到 (2,)
        probs = self.pi(state).squeeze(0)
        # 根据动作概率分布π_θ(A_t|S_t)采样，得到动作a_t
        # torch.multinomial：按 probs 做多项式分布采样，num_samples=1 表示采 1 个样本
        # .item()：把只含 1 个元素的张量转成 Python int
        action = torch.multinomial(probs, num_samples=1).item()
        # 返回值说明：
        # action: 采样到的具体动作（0 或 1）
        # probs: 动作的概率分布（形状 (2,)，后续 REINFORCE 计算损失时要用到）
        return action, probs

    def rollout(self, env):
        """在环境 env 中采样一条轨迹（即玩一个回合的游戏）：从初始状态 S0 开始，
        按当前策略不断采样动作并执行，直到回合结束（done=True），
        返回轨迹 τ = (S0, A0, R0, S1, A1, R1, ..., S_{T-1}, A_{T-1}, R_{T-1}) 对应的三个列表"""
        state = env.reset()  # 初始状态 S0
        # tau = (S0, A0, R0, S1, A1, R1, ...)
        states = []  # [S0, S1, ...] 每一步的状态列表
        actions = []  # [A0, A1, ...] 每一步采取的动作列表
        rewards = []  # [R0, R1, ...] 每一步的即时奖励列表

        done = False
        while not done:
            # 按当前策略 π_θ 采样一个动作 a_t（probs 这里暂不收集）
            action, _ = self.get_action(state)
            # 在环境中执行动作，环境返回下一步状态、即时奖励和终止标志
            next_state, reward, done, _ = env.step(action)

            # 保存本步的轨迹信息（只存 t 时刻的 state/action/reward，终止状态 S_T 不存储）
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # 环境转移到下一个状态
            state = next_state

        return states, actions, rewards


agent = Agent()  # 实例化智能体（策略网络随机初始化）
states, actions, rewards = agent.rollout(env)  # 在环境中采样一条完整轨迹

# 逐条打印轨迹：每个时刻的状态、采取的动作、获得的即时奖励
for s, a, r in zip(states, actions, rewards):
    print(f"状态：{s}, 采取的动作：{a}, 获得的即时奖励：{r}")

# 计算一条轨迹的带折扣因子的回报 G_0 = R_0 + γR_1 + γ²R_2 + ...
G = 0.0  # 从 G_T = 0 开始逆序累加（回合终止后没有后续奖励）
# 递推关系（从后往前算）：
# G_T = 0（终止后没有奖励；原注释 G_T = R_T 不准确，已修正）
# G_{T-1} = R_{T-1} + γG_T  # 原式保留：这是递推式 G_t = R_t + γG_{t+1} 在 t=T-1 时的特例
# ...
for r in rewards[::-1]:  # 逆序遍历奖励序列：从最后一步 R_{T-1} 一路算回 R_0
    # G_t = R_t + γG_{t+1}
    G = r + agent.gamma * G  # 当前步的折扣回报 = 即时奖励 + γ × 后续累计回报

print(f"带折扣因子的回报: {G}")  # 未训练策略下该值波动较大；训练后的策略应能获得更高回报
