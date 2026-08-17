# 演员评论家（单步TD误差）策略梯度法
# =============================================================================
# 算法简介：Actor-Critic（演员-评论家，单步 TD 误差版本）
# -----------------------------------------------------------------------------
# Actor-Critic 在策略梯度法的基础上引入两个网络，分别扮演两个角色：
#   演员 Actor：策略网络 π_θ(a|s)，负责选择动作（对应 PolicyNet）
#   评论家 Critic：价值网络 V_ω(s)，负责评估状态的好坏（对应 ValueNet）
#
# 核心思想：用评论家给出的单步 TD 误差 δ_t 替代 REINFORCE 中的蒙特卡洛回报
# G_t，作为策略梯度的加权项（δ_t 可视为优势函数的无偏估计，衡量"这一步做得
# 比评论家预期好多少"）：
#
#   TD 目标：  y_t = R_t + γ·V_ω(S_{t+1})             （终止状态 S_T 处视 V=0）
#   TD 误差：  δ_t = y_t - V_ω(S_t)                   （评论家对当前状态的"打分偏差"）
#   演员梯度： ∇J(θ) ≈ Σ_t δ_t · ∇_θ log π_θ(A_t|S_t) （δ_t>0 则增大该动作概率，反之减小）
#   评论家更新：ω ← ω - α_v · ∇_ω (y_t - V_ω(S_t))²   （让 V_ω 拟合 TD 目标）
#
# 与 REINFORCE 的区别（bias-variance 权衡）：
#   REINFORCE 用整条轨迹的蒙特卡洛回报 G_t，无偏但方差大，且必须等回合结束；
#   Actor-Critic 用单步 TD 误差 δ_t，方差小、可逐步更新，但 V_ω 本身是近似值，
#   因此 δ_t 是有偏估计。
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
    """策略神经网络 π_θ(A_t|S_t)——演员（Actor）。

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
        x = self.l1(x)              # 线性层 4 → 128，shape: (B, 128)
        x = F.relu(x)               # ReLU 非线性激活，shape: (B, 128)
        x = self.l2(x)              # 线性层 128 → 2，此时输出的是 logits，shape: (B, 2)
        x = F.softmax(x, dim=-1)    # softmax 把 logits 归一化为概率分布（两个动作概率和为 1），shape: (B, 2)
        return x


class ValueNet(nn.Module):
    """价值神经网络 V_ω(S_t)——评论家（Critic）。

    输入状态 S_t，输出该状态的估计价值（标量）。
    网络结构与 PolicyNet 类似，但输出层只有 1 个神经元，
    且最后一层不加激活函数：状态价值可以是任意实数，无需归一化。
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)  # 输入层：状态是 4 个浮点数
        self.l2 = nn.Linear(128, 1)  # 输出层：输出 1 个标量 V_ω(S)

    def forward(self, x):
        """前向传播。

        参数:
            x: 一批状态，shape: (B, 4)
        返回:
            每个状态的估计价值，shape: (B, 1)
        """
        x = self.l1(x)      # 线性层 4 → 128，shape: (B, 128)
        x = F.relu(x)       # ReLU 非线性激活，shape: (B, 128)
        x = self.l2(x)      # 线性层 128 → 1，输出标量价值，shape: (B, 1)
        return x


class Agent:
    """智能体：封装演员/评论家两个网络、动作选择、轨迹采样与参数更新。"""

    def __init__(self):
        self.gamma = 0.98  # 折扣因子 gamma：越远的未来奖励权重越小（γ^k 随 k 衰减）
        self.pi = PolicyNet()  # 策略网络（演员）
        self.v = ValueNet()  # 价值网络（评论家）
        self.lr_pi = 0.001  # 演员（策略网络）的学习率
        self.lr_v = 0.05  # 评论家（价值网络）的学习率，通常设得比演员大，让价值估计尽快跟上
        self.optimizer_pi = torch.optim.Adam(  # 演员的 Adam 优化器
            self.pi.parameters(),
            lr=self.lr_pi
        )
        self.optimizer_v = torch.optim.Adam(  # 评论家的 Adam 优化器（两个网络各自独立的优化器）
            self.v.parameters(),
            lr=self.lr_v
        )

    def get_action(self, state):
        """按策略网络输出的概率分布随机采样一个动作。

        参数:
            state: 当前状态 S_t，shape: (4,)
        返回:
            action: 采样到的具体动作（0 或 1）
            probs:  策略网络输出的动作概率分布（当前代码未使用，保留便于调试或扩展）
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
        """在环境 env 中采样一条轨迹 τ（玩一个回合的游戏），并计算 TD 目标与 TD 误差。

        轨迹 τ = (S0, A0, R1, S1, A1, R2, ..., S_T)。
        与 REINFORCE 不同：单步 TD 需要下一状态 S_{t+1} 和终止标志，
        因此额外收集 next_states 和 dones。

        返回:
            states:     [S0, ..., S_{T-1}]，shape: (B, 4)
            actions:    [A0, ..., A_{T-1}]，shape: (B, 1)
            rewards:    [R0, ..., R_{T-1}]，shape: (B, 1)（update 中未使用，仅用于统计监控）
            td_targets: 每个时间步的 TD 目标 y_t，shape: (B, 1)
            td_errors:  每个时间步的 TD 误差 δ_t，shape: (B, 1)
        """
        state = env.reset()  # 重置环境，得到初始状态 S0
        # 用于记录整条轨迹的数据
        states = []  # [S0, S1, ..., S_{T-1}]
        next_states = []  # [S1, S2, ..., S_T]（TD 目标需要下一状态）
        actions = []  # [A0, A1, ..., A_{T-1}]
        rewards = []  # [R0, R1, ..., R_{T-1}]
        dones = []  # [False, False, ..., True]（最后一个时间步为 True，表示回合结束）
        done = False  # 回合结束标志

        while not done:
            # ① 按当前策略 π_θ 选择一个动作
            action, _ = self.get_action(state)
            # ② 在环境中执行该动作，得到下一状态、即时奖励与结束标志
            next_state, reward, done, _ = env.step(action)

            # ③ 保存这一步轨迹的信息
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            # ④ 环境转移到下一个状态，继续循环
            state = next_state

        # ---- 把各列表转为 tensor，便于向量化计算 ----
        states = torch.tensor(states)  # (B, 4)
        next_states = torch.tensor(next_states)  # (B, 4)
        # [0, 1, 0, 1, ...] --> [[0],[1],[0],[1],...]
        # 形状变换是：(B,) --> (B, 1)，为后续 gather 和逐元素运算对齐形状
        actions = torch.tensor(actions).view(-1, 1)
        rewards = torch.tensor(rewards).view(-1, 1)  # (B, 1)
        # 布尔值转为 float（True→1.0，False→0.0），用于屏蔽终止状态后的价值
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)  # (B, 1)

        # ---- 计算单步 TD 目标：y_t = R_t + γ·V_ω(S_{t+1}) ----
        # 对每个时间步：[[R_0 + γV_ω(S_1)],[R_1 + γV_ω(S_2)],...,[R_{T-1}]]
        # (1 - dones)：若 S_{t+1} 是终止状态（done=True），则 V_ω(S_{t+1}) 视为 0
        # （回合结束后没有未来回报，与 REINFORCE 中 G 的递推初始化为 0 是同一约定）
        td_targets = rewards + self.gamma * self.v(next_states) * (1 - dones)
        # TD 目标是"拟合的目标"，作为常数使用：detach 切断梯度，
        # 使价值网络的更新只发生在 loss_v 处，不通过 TD 目标反向传播
        td_targets = td_targets.detach()

        # ---- 计算单步 TD 误差：δ_t = y_t - V_ω(S_t) ----
        # 衡量"实际回报 + 下一状态价值"比评论家对当前状态的估计好多少
        td_errors = td_targets - self.v(states)
        # 同理 detach：δ_t 在演员更新中只作为加权系数，本身不参与梯度传播
        # （演员的梯度只从 log π_θ 处回传，δ_t 相当于常量权重）
        td_errors = td_errors.detach()

        return states, actions, rewards, td_targets, td_errors

    def update(self, trajectory):
        """使用一条轨迹分别更新演员（策略网络）和评论家（价值网络）的参数。

        ① 演员：loss_pi = -Σ_t δ_t · log π_θ(A_t|S_t)
            最小化 loss_pi 等价于梯度上升：δ_t>0 的动作概率增大，δ_t<0 的减小。
        ② 评论家：loss_v = MSE(V_ω(S_t), y_t)
            让价值网络拟合 TD 目标，缩小 TD 误差。

        参数:
            trajectory: rollout 的返回值 (states, actions, rewards, td_targets, td_errors)
        """
        states, actions, rewards, td_targets, td_errors = trajectory

        # ---- ① 计算策略网络（演员）的损失 ----
        # 取出每个时间步实际执行动作对应的概率再取对数：
        # gather 沿 dim=1 按 actions 中的索引提取 π_θ(A_t|S_t)，
        # 结果 shape: (B, 1)，即 [[logπ_θ(A_0|S_0)], ..., [logπ_θ(A_{T-1}|S_{T-1})]]
        log_probs = torch.log(self.pi(states).gather(1, actions))
        # 目标函数：obj = Σ_t δ_t · log π_θ(A_t|S_t)
        # TD 误差 δ_t 作为加权系数：这一步做得比评论家预期好（δ_t>0），
        # 就增大该动作的对数概率；反之减小
        obj = torch.sum(log_probs * td_errors)

        # 取负号：深度学习框架习惯做梯度下降，
        # minimize loss_pi = -obj 等价于 maximize obj（梯度上升）
        loss_pi = -obj

        # ---- ② 计算价值网络（评论家）的损失 MSE----
        # 均方误差：让 V_ω(S_t) 逼近 TD 目标 y_t（td_targets 已 detach，是常量）
        loss_v = F.mse_loss(self.v(states), td_targets)

        # ---- 分别更新两个网络（先清各自优化器的梯度，再反向传播，最后各更新各的参数）----
        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()
        loss_pi.backward()  # 演员的梯度：只流经策略网络
        loss_v.backward()  # 评论家的梯度：只流经价值网络
        self.optimizer_pi.step()
        self.optimizer_v.step()


# ---- 主流程：与环境交互，训练 3000 个回合 ----
agent = Agent()  # 实例化智能体
env = gym.make("CartPole-v0")  # 创建 CartPole 环境

episodes = []  # 记录回合编号（画图的 x 轴）
returns = []  # 记录每个回合的总回报（画图的 y 轴）

for episode in range(3000):
    # ① 采样轨迹：用当前策略 π_θ 在环境中玩一个回合，并计算 TD 目标与 TD 误差
    tau = agent.rollout(env)
    # ② 更新：分别更新演员（策略网络）和评论家（价值网络）
    agent.update(tau)

    # 记录本回合数据（总回报 = 该回合所有即时奖励之和，即本回合坚持的步数）
    # sum 对 (B,1) 的 tensor 求和后仍是 tensor，取 .item() 转为 Python 标量
    episodes.append(episode)
    returns.append(sum(tau[2]).item())

    # 从后往前递推整条轨迹的折扣总回报 G(τ)，仅用于监控对比
    G = 0.0
    # 把 rewards 转为 Python 列表并倒序，递推 G = r + γ·G
    for r in tau[2].squeeze(-1).numpy().tolist()[::-1]:
        G = r + 0.98 * G  # 注：折扣因子硬编码为 0.98，与 agent.gamma 一致；建议引用 agent.gamma 避免两处不一致

    # 每 100 个回合打印一次当前表现：
    # Return 为本回合总回报；G(τ) 为折扣总回报；
    # V_ω(S0) 由 td_targets[0] - td_errors[0] 还原（两者之差即 V(states[0])），
    # 即评论家对本回合初始状态的估值（注意：这是本回合更新前的估值）
    if (episode + 1) % 100 == 0:
        print(
            f"Episode: {episode+1}, Return: {sum(tau[2]).item()}, G: {G}, V_ω(S0): {(tau[3] - tau[4])[0].item()}")


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


plot_loss(episodes, returns, "actor-critic-pg-loss.pdf")  # 绘制并保存学习曲线