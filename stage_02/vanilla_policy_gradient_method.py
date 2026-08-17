# 原始（vanilla）策略梯度法
# =============================================================================
# 算法简介：原始（vanilla）策略梯度法
# -----------------------------------------------------------------------------
# 直接参数化策略 π_θ(a|s)（本文件用神经网络表示），通过采样轨迹估计策略
# 梯度并做梯度上升，使高回报动作的采样概率逐渐增大：
#
#   精确梯度：∇J(θ) = E_π[ Σ_t ∇_θ log π_θ(A_t | S_t) · R(τ) ]
#   蒙特卡洛近似（对每条轨迹 τ）：∇J(θ) ≈ G(τ) · Σ_t ∇_θ log π_θ(A_t | S_t)
#   参数更新：θ ← θ + α · ∇J(θ)
#   实现等价于对 loss = -G(τ) · Σ_t log π_θ(A_t|S_t) 做梯度下降（取负号变上升为下降）
#
# 其中 G(τ) = Σ_t γ^t · R_t 是整条轨迹的折扣总回报。
# 本方法的特征：轨迹内所有时间步共用同一个回报 G(τ) 作为权重（即轨迹的总回报
# 对整条轨迹上每个动作的概率统一加权），是最朴素的策略梯度形式。
#
# 与 REINFORCE 的区别（见 reinforce-policy-gradient-method.py）：
#   vanilla（本文件）：用整条轨迹统一的折扣总回报 G(τ) 加权所有时间步；
#   REINFORCE：用每个时间步"从该时刻起算"的折扣回报
#              G_t = Σ_{k=t}^{T} γ^(k-t) · R_k 分别加权各时间步。
#
# 环境：CartPole-v0（倒立摆小车）
#   状态 S：4 维连续量（小车位置、小车速度、杆角度、杆角速度）
#   动作 A：2 个离散动作（0 = 向左推，1 = 向右推）
#   奖励 R：每坚持一步 +1；杆倾倒或小车滑出边界则回合结束
# =============================================================================

import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """策略神经网络 π_θ(A_t|S_t)。

    输入状态 S_t，输出该状态下各个动作的采样概率分布。
    网络结构：4（状态维）→ 128（隐藏层）→ 2（动作数），最后一层接 softmax。
    """

    def __init__(self):
        super().__init__()
        # 输入层：状态是 4 个浮点数（CartPole 的状态维度）
        self.l1 = nn.Linear(4, 128)
        # 输出层：动作数量为 2 个（CartPole 只有"左推/右推"两个动作）
        self.l2 = nn.Linear(128, 2)

    def forward(self, x):
        """前向传播。

        参数:
            x: 一批状态，shape: (B, 4)
        返回:
            各动作的采样概率分布，shape: (B, 2)
        """
        x = self.l1(x)  # 线性层 4 → 128，shape: (B, 128)
        x = F.relu(x)  # ReLU 非线性激活，shape: (B, 128)
        x = self.l2(x)  # 线性层 128 → 2，此时输出的是 logits，shape: (B, 2)
        x = F.softmax(x, dim=-1)  # softmax 把 logits 归一化为概率分布（两个动作概率和为 1），shape: (B, 2)
        return x


class Agent:
    """智能体：封装策略网络、动作选择、轨迹采样与策略更新（vanilla 策略梯度主体）。"""

    def __init__(self):
        self.gamma = 0.98  # 折扣因子 gamma：越远的未来奖励权重越小（γ^k 随 k 衰减）
        self.pi = PolicyNet()  # 策略网络 π_θ
        self.lr_pi = 0.001  # 策略网络的学习率
        self.optimizer_pi = torch.optim.Adam(  # Adam 优化器（自适应学习率的梯度下降变体）
            self.pi.parameters(),
            lr=self.lr_pi
        )

    def get_action(self, state):
        """按策略网络输出的概率分布随机采样一个动作。

        参数:
            state: 当前状态 S_t，shape: (4,)
        返回:
            action: 采样到的具体动作（0 或 1）
            probs:  策略网络输出的动作概率分布（当前代码未使用，保留便于调试或扩展）

        """
        """
            环境state：                  (4,)        → [s0,s1,s2,s3]
            unsqueeze(0)                (1,4)       → [[s0,s1,s2,s3]]   # 适配网络batch输入
            self.pi(state) 网络推理      (1,2)       → [[0.6,0.4]]
            squeeze(0)                  (2,)        → [0.6,0.4]         # 拆掉batch维，用于采样
        """

        # 将状态从形状 (4,) 变成 (1, 4)：增加 batch 维度以匹配网络的输入格式 (B, 4)
        state = torch.tensor(state).unsqueeze(0)
        # 前向传播得到各动作的采样概率，再去掉 batch 维度，shape: (2,)
        probs = self.pi(state).squeeze(0)
        # 按概率分布采样 1 个动作（多项式分布采样，即"依概率掷骰子"），并取出标量值
        # 注意：这里是随机采样而非取 argmax，保证智能体能探索到新的动作
        action = torch.multinomial(
            probs,
            num_samples=1,
        ).item()
        # action: 采样到的具体动作
        # probs:  动作的概率分布
        return action, probs

    def rollout(self, env):
        """在环境 env 中采样一条轨迹 τ（玩一个回合的游戏）。

        轨迹 τ = (S0, A0, R1, S1, A1, R2, ...)，本方法按三个列表分别保存。

        返回:
            states:  [S0, S1, ...] 各时间步的状态
            actions: [A0, A1, ...] 各时间步执行的动作
            rewards: [R1, R2, ...] 各时间步获得的即时奖励
        """
        state = env.reset()  # 重置环境，得到初始状态 S0
        # 用于记录整条轨迹的数据
        states = []  # [S0, S1, ...]
        actions = []  # [A0, A1, ...]
        rewards = []  # [R1, R2, ...]

        done = False  # 回合结束标志

        while not done:
            # ① 按当前策略 π_θ 选择一个动作
            action, _ = self.get_action(state)
            # ② 在环境中执行该动作，得到下一状态、即时奖励与结束标志
            next_state, reward, done, _ = env.step(action)

            # ③ 保存这一步轨迹的信息
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # ④ 环境转移到下一个状态，继续循环
            state = next_state

        return states, actions, rewards

    def update(self, trajectory):
        """使用一条轨迹做一次 vanilla 策略梯度更新。

        目标函数：obj = Σ_t G(τ) · log π_θ(A_t|S_t) = G(τ) · Σ_t log π_θ(A_t|S_t)，
        即用整条轨迹的折扣总回报 G(τ) 对所有时间步统一加权。
        实际优化 loss = -obj：最小化 loss 等价于最大化 obj（增大高回报动作的对数
        概率），即对策略梯度做梯度上升。

        参数:
            trajectory: (states, actions, rewards) 三元组，即 rollout 的返回值
        """
        states, actions, rewards = trajectory

        # ---- 计算整条轨迹的折扣总回报 G(τ) = Σ_t γ^t · R_t ----
        # 从后往前迭代累加，循环结束后 G 即整条轨迹的折扣总回报
        G = 0.0  # 计算 G(τ)
        for r in rewards[::-1]:
            # 递推式 G = r + γ·G 从后往前展开：
            # 最终得到 G(τ) = R1 + γ·R2 + γ²·R3 + ... = Σ_t γ^t · R_t
            G = r + self.gamma * G
        # 注意：与 REINFORCE 不同，这里只保留循环结束后的最终值，
        # 轨迹内所有时间步共用同一个轨迹总回报 G(τ)，而非各自的 G_t

        # ---- 计算目标函数 obj = G(τ) · Σ_t log π_θ(A_t|S_t) ----
        obj = 0.0
        for s, a in zip(states, actions):
            # 计算 log π_θ(A_t|S_t)：把状态过一遍策略网络，
            # 取出实际执行动作 a 对应的概率，再取对数
            s = torch.tensor(s).unsqueeze(0)
            log_action_prob = torch.log(self.pi(s).squeeze(0)[a])
            # 累加 G(τ)·log π：同一个轨迹总回报对所有时间步统一加权，
            # 轨迹回报越高，整条轨迹上所有动作的概率都被同比例增大
            obj += log_action_prob * G

        # 取负号：深度学习框架习惯做梯度下降，
        # minimize loss = -obj 等价于 maximize obj（梯度上升）
        loss = -obj

        # 标准三步梯度更新：清空梯度 → 反向传播求 ∇loss → 更新参数
        self.optimizer_pi.zero_grad()
        loss.backward()
        self.optimizer_pi.step()


# ---- 主流程：与环境交互，训练 3000 个回合 ----
agent = Agent()  # 实例化智能体
env = gym.make("CartPole-v0")  # 创建 CartPole 环境

episodes = []  # 记录回合编号（画图的 x 轴）
returns = []  # 记录每个回合的总回报（画图的 y 轴）

for episode in range(3000):
    # ① 采样轨迹：用当前策略 π_θ 在环境中玩一个回合，得到轨迹 τ
    tau = agent.rollout(env)
    # ② 更新策略神经网络：用这条轨迹估计策略梯度并更新参数 θ
    agent.update(tau)

    # 记录本回合数据（总回报 = 该回合所有即时奖励之和，即本回合坚持的步数）
    episodes.append(episode)
    returns.append(sum(tau[2]))

    # 每 100 个回合打印一次当前表现，观察训练进展
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Return: {sum(tau[2])}")


def plot_loss(episodes, returns, filename):
    """绘制并保存"回合数-总回报"学习曲线。

    参数:
        episodes: x 轴数据（回合编号）
        returns:  y 轴数据（每回合总回报）
        filename: 保存的图片文件名
    """
    f = plt.figure()  # 新建画布
    plt.plot(episodes, returns)  # 绘制折线图
    plt.xlabel("Episodes")  # x 轴标签
    plt.ylabel("Returns")  # y 轴标签
    plt.title("CartPole-v0")  # 图标题
    plt.show()  # 显示图像
    # 注：建议把 savefig 移到 show() 之前；在部分非交互式后端下，
    # show() 之后图像对象可能已被清空，导致保存的图片为空
    f.savefig(filename, bbox_inches="tight")  # 保存为 pdf 文件


plot_loss(episodes, returns, "pg-loss.pdf")  # 绘制并保存学习曲线
