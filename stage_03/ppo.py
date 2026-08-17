"""
PPO (Proximal Policy Optimization) —— 近端策略优化算法
基于演员-评论家(Actor-Critic)架构,使用广义优势估计(GAE)与裁剪目标函数

核心思想:
  1. 策略网络(演员 Actor)输出动作概率分布、负责采样动作;
     价值网络(评论家 Critic)估计状态价值,为优势计算提供基线
  2. GAE 对多步 TD 误差做折扣加权组合:方差小于蒙特卡洛回报,
     偏差小于单步 TD 误差,在两者间取得平衡
  3. 裁剪目标函数将新旧策略的概率比限制在 [1-ε, 1+ε] 内,
     避免单步更新过大导致策略性能崩溃

训练流程(每次迭代):
  rollout → 按当前策略交互一整条轨迹(一个 episode),计算 GAE 优势
  update  → 用同一批轨迹内层循环多次更新策略与价值网络(提高样本利用率)

参考:Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """策略网络 π_θ(A_t | S_t):输入状态,输出各动作的概率分布

    网络结构:状态(4) → 128(ReLU) → 2(SoftMax)
    - 输入维度 4:CartPole 的状态空间 [小车位置, 小车速度, 杆角度, 杆角速度]
    - 输出维度 2:CartPole 的动作空间 [向左推, 向右推]
    - 最后一层 SoftMax 保证输出为合法概率分布(各分量 ≥ 0 且和为 1)
    - 网络仅负责输出分布,具体动作由 Agent.get_action 按该分布采样决定
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)    # 状态编码层:4维状态 → 128维隐藏表示
        self.l2 = nn.Linear(128, 2)   # 动作打分层:128维隐藏 → 2个动作的 logit(未归一化得分)

    def forward(self, x):
        """前向传播

        Args:
            x: 状态张量,shape (B, 4),B 为批量大小

        Returns:
            动作概率分布,shape (B, 2),每行是一个合法概率分布
        """
        x = self.l1(x)                  # (B, 4) → (B, 128),线性变换
        x = F.relu(x)                   # (B, 128) → (B, 128),ReLU 激活引入非线性
        x = self.l2(x)                  # (B, 128) → (B, 2),输出各动作的 logit
        x = F.softmax(x, dim=-1)        # (B, 2) → (B, 2),SoftMax 归一化为概率
        return x


class ValueNet(nn.Module):
    """价值网络 V_ω(S_t):输入状态,输出该状态的价值估计(期望累计折扣回报)

    网络结构:状态(4) → 128(ReLU) → 1
    - 输入维度 4:与策略网络相同的状态空间
    - 输出维度 1:标量价值,即 V_ω(S_t) ≈ E_π[Σ_k γ^k R_{t+k}],
      期望关于当前策略 π 与环境转移的分布
    - 无 SoftMax:价值是任意实数,不需要归一化

    注:价值网络作为基线(预测折扣回报),用于计算 TD 误差与 GAE 优势;
        基线选取得当可显著降低策略梯度估计的方差
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)    # 状态编码层:与策略网络结构相同
        self.l2 = nn.Linear(128, 1)   # 价值输出层:128维隐藏 → 1维标量价值

    def forward(self, x):
        """前向传播

        Args:
            x: 状态张量,shape (B, 4)

        Returns:
            状态价值估计,shape (B, 1)
        """
        x = self.l1(x)                  # (B, 4) → (B, 128)
        x = F.relu(x)                   # (B, 128) → (B, 128)
        x = self.l2(x)                  # (B, 128) → (B, 1)
        return x


class Agent:
    """PPO 智能体:封装策略网络、价值网络及其交互与更新逻辑

    关键超参数:
    - gamma (γ):折扣因子,控制未来奖励的衰减速度,γ 越大越重视长期回报
    - lmbda (λ):GAE 超参数,控制偏差-方差权衡
        λ=0 → 仅用单步 TD 误差(偏差大、方差小)
        λ=1 → 用完整轨迹的蒙特卡洛回报(偏差小、方差大)
        典型值 0.95 是一个较好的折中
    - lr_pi:策略网络学习率,通常较小(0.001)以避免策略更新过激
    - lr_v:价值网络学习率,可以较大(0.05)因为价值回归更稳定
    - epsilon (ε):裁剪阈值,限制新旧策略概率比,update 中使用 ε=0.2
    - N:内层更新次数,同一条轨迹上重复利用 10 次
    """

    def __init__(self):
        self.gamma = 0.98              # 折扣因子 γ:未来奖励的衰减系数
        self.lmbda = 0.95              # GAE 超参数 λ:偏差-方差权衡
        self.pi = PolicyNet()          # 策略网络(演员 Actor)
        self.v = ValueNet()            # 价值网络(评论家 Critic)
        self.lr_pi = 0.001             # 策略网络学习率
        self.lr_v = 0.05               # 价值网络学习率
        # 两个网络各自独立的 Adam 优化器,参数更新互不干扰
        self.optimizer_pi = torch.optim.Adam(
            self.pi.parameters(),
            lr=self.lr_pi
        )
        self.optimizer_v = torch.optim.Adam(
            self.v.parameters(),
            lr=self.lr_v
        )

    def get_action(self, state):
        """根据当前策略采样一个动作

        Args:
            state: 环境状态,shape (4,),numpy 数组

        Returns:
            action: 采样的动作索引(整数 0 或 1)
            probs: 各动作的概率分布,shape (2,)
        """
        # 将 numpy 状态转为张量,增加 batch 维度:(4,) → (1, 4)
        state = torch.tensor(state).unsqueeze(0)
        # 前向传播得到概率分布,去掉 batch 维度:(1, 2) → (2,)
        probs = self.pi(state).squeeze(0)
        # 按概率分布采样一个动作(加权随机选择)
        # 刻意不用 argmax:随机采样保持探索性(exploration),
        # 让低概率动作仍有机会被尝试,这也是策略梯度学习随机策略的基础
        action = torch.multinomial(
            probs,
            num_samples=1,
        ).item()
        return action, probs

    def rollout(self, env):
        """在环境中采样一条完整轨迹,并计算 GAE 优势与旧策略对数概率

        完整流程:
        1. 用当前策略在 env 中交互至回合结束,收集 (S, A, R, S') 序列
        2. 计算单步 TD 误差:δ_t = R_t + γ V_ω(S_{t+1}) - V_ω(S_t)
        3. 从后往前递推 GAE 优势:A_t = δ_t + (γλ) A_{t+1}
        4. 计算价值网络回归目标:target_t = A_t + V_ω(S_t)
        5. 记录旧策略下动作的对数概率 log π_θ_old(A_t|S_t),供 update 中的
           重要性采样比率 r_t = exp(log π_new - log π_old) 使用

        Args:
            env: gym 环境

        Returns:
            states:        所有时刻的状态,shape (T, 4)
            actions:       所有时刻的动作,shape (T, 1)
            rewards:       所有时刻的奖励,shape (T, 1)
            gae_targets:   价值网络回归目标,shape (T, 1),已 detach
            gae_errors:    GAE 优势估计值,shape (T, 1)
            old_log_probs: 旧策略下动作的对数概率,shape (T, 1),已 detach
        """
        state = env.reset()  # 重置环境,获取初始状态 S0

        # 收集轨迹数据(列表形式,长度为 T,即回合步数)
        states = []          # [S_0, S_1, ..., S_{T-1}]
        next_states = []     # [S_1, S_2, ..., S_T]
        actions = []         # [A_0, A_1, ..., A_{T-1}]
        rewards = []         # [R_0, R_1, ..., R_{T-1}]
        dones = []           # [False, False, ..., True](回合结束标志)

        done = False

        while not done:
            # 根据当前策略采样动作
            action, _ = self.get_action(state)
            # 在环境中执行动作,返回 (下一状态, 奖励, 是否终止, info 诊断信息)
            next_state, reward, done, _ = env.step(action)

            # 保存当前转移到轨迹中
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            # 推进到下一个状态
            state = next_state

        # --- 将列表转为张量,统一形状便于后续批量计算 ---
        states = torch.tensor(states)                                    # (T, 4)
        next_states = torch.tensor(next_states)                          # (T, 4)
        actions = torch.tensor(actions).view(-1, 1)                      # (T,) → (T, 1)
        rewards = torch.tensor(rewards).view(-1, 1)                      # (T,) → (T, 1)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)       # (T,) → (T, 1),标记回合是否终止,用于截断后继价值

        # --- 计算单步 TD 误差 ---
        # δ_t = R_t + γ V_ω(S_{t+1}) · (1 - done_t) - V_ω(S_t)
        # done_t = True 时乘 0 截断:回合已终止,后继状态 S_{t+1} 不存在,
        # 其价值不应计入回报,故终止步的 TD 误差为 δ_T = R_T - V_ω(S_T)
        td_errors = rewards + self.gamma * \
            self.v(next_states) * (1 - dones) - self.v(states)
        # detach:TD 误差在此处仅作为 GAE 的输入数据,不参与梯度回传
        # squeeze + tolist:转为 Python 列表,便于后续逆序遍历
        td_errors = td_errors.detach().squeeze(-1).tolist()

        # --- 递推计算广义优势估计 (Generalized Advantage Estimation, GAE) ---
        # GAE 公式(从后往前递推):
        #   A_t = δ_t + γλ · A_{t+1}
        # 等价于展开形式:
        #   A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ... + (γλ)^{T-t-1} δ_{T-1}
        # λ=0 时退化为单步 TD 误差 A_t = δ_t(偏差大,方差小)
        # λ=1 时退化为蒙特卡洛回报(偏差小,方差大)
        lastgae = 0.0          # A_{t+1},初始为 0(最后一个时刻之后无优势)
        gae_errors = []        # 存放各时刻的 GAE 优势值
        for delta in td_errors[::-1]:   # 从最后一个时刻向前遍历
            lastgae = delta + self.gamma * self.lmbda * lastgae
            gae_errors.insert(0, lastgae)   # insert(0, ·) 保持时间正序

        gae_errors = torch.tensor(gae_errors).view(-1, 1)    # (T,) → (T, 1)

        # --- 计算价值网络的回归目标 ---
        # GAE 优势 A_t 是对"该动作比平均水平好多少"的估计
        # 价值目标 = A_t + V_ω(S_t),即 V_target ≈ V_ω(S_t) + A_t
        # 这等价于用 TD(λ) 回报来训练价值网络,比单步 TD 目标更稳定
        gae_targets = gae_errors + self.v(states)
        gae_targets = gae_targets.detach()   # detach:目标值不应参与梯度计算,否则会导致梯度不稳定

        # --- 记录旧策略下各动作的对数概率 ---
        # log π_θ_old(A_t | S_t),用于 PPO 裁剪目标中的重要性采样比值
        # gather(1, actions):从概率分布中选取实际执行动作对应的概率
        # detach:旧策略概率是固定的参考值,不参与梯度更新
        old_log_probs = torch.log(self.pi(states).gather(1, actions)).detach()

        return states, actions, rewards, gae_targets, gae_errors, old_log_probs

    def update(self, trajectory):
        """使用 PPO 裁剪目标函数更新策略网络和价值网络

        PPO 的核心是裁剪代理目标函数(Clipped Surrogate Objective):
          L^CLIP(θ) = E_t [ min( r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]

        其中:
          r_t(θ) = π_θ(A_t|S_t) / π_θ_old(A_t|S_t)  是新旧策略的概率比
          A_t 是 GAE 优势估计
          ε 是裁剪范围(本实现中 ε=0.2)

        裁剪的作用:
          - 当 A_t > 0(好动作):r_t 被限制在 [0, 1+ε],防止过度增大概率
          - 当 A_t < 0(差动作):r_t 被限制在 [1-ε, +∞],防止过度减小概率
          - 无论哪种情况,策略更新幅度都被约束在合理范围内

        内层循环(N=10):
          PPO 的核心创新是用多次小步更新替代 TRPO 的一次大步更新,
          在同一条轨迹上重复利用数据;裁剪目标函数保证每步更新
          都不会偏离旧策略太远,从而保证数据复用是安全的

        Args:
            trajectory: rollout() 返回的元组
                (states, actions, rewards, gae_targets, gae_errors, old_log_probs)
        """
        # 解包 rollout() 返回的轨迹数据
        states, actions, rewards, gae_targets, gae_errors, old_log_probs = trajectory

        # PPO 内层循环:在同一条轨迹上多次更新,提高数据利用效率
        # 原始策略梯度法只用一次就丢弃数据,PPO 通过裁剪安全地复用多次
        for _ in range(10):   # 内层更新次数 N=10(超参数,见类文档)
            # --- ① 策略网络损失(PPO 裁剪目标) ---

            # 计算新策略下各动作的对数概率
            # gather(1, actions):从 (T, 2) 的概率矩阵中选取实际动作对应的概率
            log_probs = torch.log(self.pi(states).gather(1, actions))   # (T, 1)

            # 新旧策略的概率比 r_t(θ) = π_θ(A_t|S_t) / π_θ_old(A_t|S_t)
            # 利用对数差计算:r = exp(log π_new - log π_old),数值上更稳定
            ratio = torch.exp(log_probs - old_log_probs)                # (T, 1)

            # 裁剪概率比:将 r_t 限制在 [1-ε, 1+ε] 范围内,ε=0.2
            # 当 r_t 在 [0.8, 1.2] 内时,clip 值 = r_t(无裁剪)
            # 当 r_t 超出范围时,clip 值 = 边界值(裁剪生效)
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)       # (T, 1)

            # PPO 目标的两项:
            # min_left  = r_t · A_t         (未裁剪的重要性采样目标)
            # min_right = clip(r_t) · A_t   (裁剪后的目标)
            # 取 min 的效果:
            #   A_t > 0 时:限制 r_t 不超过 1+ε,即不让好动作概率增加太多
            #   A_t < 0 时:限制 r_t 不低于 1-ε,即不让差动作概率减少太多
            min_left = ratio * gae_errors
            min_right = clipped_ratio * gae_errors

            # PPO 裁剪代理目标:对所有时刻取 min 后求和
            # 最大化该目标 → 策略损失 = -obj(取负号转为最小化问题)
            obj = torch.sum(torch.min(min_left, min_right))
            loss_pi = -obj

            # --- ② 价值网络损失(均方误差回归) ---
            # 训练 V_ω(S_t) 去逼近 GAE 目标值 target_t = A_t + V_ω_old(S_t)
            # 本质是让价值网络学会更准确地评估状态价值,
            # 从而在下一次 rollout 中提供更精确的 TD 误差和 GAE 优势
            loss_v = F.mse_loss(self.v(states), gae_targets)

            # --- 梯度清零 → 反向传播 → 参数更新 ---
            # 策略网络和价值网络各自独立优化,互不干扰
            self.optimizer_pi.zero_grad()
            self.optimizer_v.zero_grad()
            loss_pi.backward()       # 计算策略网络梯度
            loss_v.backward()        # 计算价值网络梯度
            self.optimizer_pi.step() # 更新策略网络参数 θ
            self.optimizer_v.step()  # 更新价值网络参数 ω


# ==================== 训练主循环 ====================

agent = Agent()
# CartPole 平衡杆任务:杆倾角超过 ±12° 或小车移出边界即终止,
# 每存活一步奖励 +1,回合最长 200 步(单回合 Return 上限 200)
# 注:CartPole-v0 已在新版 gym 中移除,可改用 CartPole-v1(规则相同)
env = gym.make("CartPole-v0")

episodes = []    # 记录每个 episode 的编号
returns = []     # 记录每个 episode 的累计奖励(未折扣)

# 训练 500 个 episode:每轮先 rollout 采样一条轨迹,再用该轨迹更新一次参数
for episode in range(500):
    # ① 用当前策略采样一条完整轨迹
    tau = agent.rollout(env)

    # ② 用 PPO 裁剪目标更新策略和价值网络
    agent.update(tau)

    # 记录本回合的累计奖励(未折扣回报 ΣR_t)
    episodes.append(episode)
    returns.append(sum(tau[2]).item())

    # 计算本回合的折扣回报 G = Σ_t γ^t R_t(从后往前递推:G = R_t + γ·G)
    # 0.98 与 agent.gamma 保持一致(教学代码直接内联该数值)
    G = 0.0
    for r in tau[2].squeeze(-1).numpy().tolist()[::-1]:
        G = r + 0.98 * G

    # 每 100 个 episode 打印一次训练指标
    if (episode + 1) % 100 == 0:
        # Return: 未折扣累计奖励(直观反映策略表现)
        # G: 折扣累计回报(理论上的优化目标,公式见上方计算)
        # V_ω(S0): 初始状态的价值估计 = gae_target - gae_error,
        #          价值网络学得好时 V(S0) ≈ G(状态价值应等于期望折扣回报)
        print(
            f"Episode: {episode+1}, Return: {sum(tau[2]).item()}, G: {G}, V_ω(S0): {(tau[3] - tau[4])[0].item()}")


def plot_loss(episodes, returns, filename):
    """绘制训练曲线:Return 随 episode 的变化

    Args:
        episodes: episode 编号列表
        returns: 每个 episode 的累计奖励列表
        filename: 保存图片的文件名

    注:函数名沿用了 loss 命名,实际绘制的是回报(Return)曲线;
       保存文件名 ppo-pg-loss.pdf 同理,仅为命名习惯,不影响内容
    """
    f = plt.figure()
    plt.plot(episodes, returns)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")   # 保存时收紧空白边距


plot_loss(episodes, returns, "ppo-pg-loss.pdf")