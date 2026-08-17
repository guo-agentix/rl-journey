# ==============================================================
# GRPO (Group Relative Policy Optimization, 分组相对策略优化)
# ==============================================================
# GRPO 与 PPO 的核心区别:
#   1. 不训练 Critic/ValueNet，只保留策略网络 PolicyNet；
#   2. 用“组内相对奖励”作为 baseline:
#        A_i = (r_i - mean(r)) / std(r)
#   3. 更新策略时仍使用 PPO-Clip，限制概率比 r_t 的变化。
#
# 本示例将一条完整 CartPole 回合看作一个“回答”:
#   每条轨迹的 reward = 本回合累计回报，再在同一组内做标准化。
# 在 LLM 场景中，G 条轨迹通常对应同一个问题下的 G 个回答，
# reward 可以是正确性、格式分或人工偏好分数。
from typing import NamedTuple

import gymnasium as gym  # MOD: gym → gymnasium，兼容 numpy 2.0+
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """策略网络 π_θ(a|s)，GRPO 只保留 Actor，不训练 ValueNet。"""

    def __init__(self, state_dim: int = 4, num_actions: int = 2):
        super().__init__()
        # CartPole 状态为 4 维，动作空间为 2 个离散动作
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回动作 logits，调用方使用 softmax 得到概率分布。"""
        return self.l2(F.relu(self.l1(x)))


def reset_env(env):
    """兼容旧版 gym 和新版 gym/gymnasium 的 reset 返回值。"""
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    """兼容旧版 gym 的 4 元组和新版 gym 的 5 元组返回值。"""
    result = env.step(action)
    if len(result) == 4:
        next_state, reward, done, _ = result
    else:
        next_state, reward, terminated, truncated, _ = result
        done = terminated or truncated
    return next_state, reward, done


class GRPOGroup(NamedTuple):
    """一组 GRPO 训练数据。

    states: (N, 4) 拼接后的状态；actions: (N, 1) 执行的动作；
    old_log_probs: (N, 1) 旧策略下实际动作的 log 概率；
    old_log_probs_dist: (N, 2) 旧策略的完整 log 概率分布；
    advantages: (N, 1) 组内相对优势；returns: (G,) 每条轨迹的累计回报。
    """

    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_log_probs_dist: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class Agent:
    """GRPO 智能体：只保留策略网络，用组内相对奖励替代 Critic。"""

    def __init__(self):
        # ---- 超参数 ----
        self.group_size = 5       # 每组采样的轨迹数量 G
        self.epsilon = 0.2        # PPO-Clip 的裁剪阈值 ε
        self.kl_coef = 0.01       # 对旧策略的 KL 惩罚系数，可设为 0
        self.update_epochs = 4    # 同一组数据内层迭代次数
        self.lr_pi = 0.001        # 策略网络学习率

        # ---- 网络与优化器 ----
        self.pi = PolicyNet()
        self.optimizer_pi = torch.optim.Adam(
            self.pi.parameters(),
            lr=self.lr_pi,
        )

    @torch.no_grad()
    def get_action(self, state) -> int:
        """从当前策略采样一个动作，采样阶段不需要反向传播。"""
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = F.softmax(self.pi(state), dim=-1).squeeze(0)
        # 按概率采样而非取 argmax，保证训练时有探索
        return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def rollout(self, env):
        """用当前策略采样一条完整轨迹，并保存旧策略的 log 概率。"""
        state = reset_env(env)
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = self.get_action(state)
            next_state, reward, done = step_env(env, action)
            states.append(torch.as_tensor(state, dtype=torch.float32))
            actions.append(action)
            rewards.append(reward)
            state = next_state

        states = torch.stack(states)
        actions = torch.as_tensor(actions, dtype=torch.long).view(-1, 1)

        # log_softmax 比“先 softmax 再取 log”更稳定；
        # gather 取出每条轨迹实际执行动作的 log 概率。
        old_log_probs_dist = F.log_softmax(self.pi(states), dim=-1)
        old_log_probs = old_log_probs_dist.gather(1, actions)

        return (
            states,
            actions,
            old_log_probs,
            old_log_probs_dist,
            float(sum(rewards)),
        )

    def sample_group(self, env) -> GRPOGroup:
        """采样 G 条轨迹，计算组内相对优势并拼接成训练批次。"""
        states_list = []
        actions_list = []
        old_log_probs_list = []
        old_log_probs_dist_list = []
        returns = []

        for _ in range(self.group_size):
            states, actions, old_log_probs, old_log_probs_dist, ret = self.rollout(env)
            states_list.append(states)
            actions_list.append(actions)
            old_log_probs_list.append(old_log_probs)
            old_log_probs_dist_list.append(old_log_probs_dist)
            returns.append(ret)

        returns = torch.as_tensor(returns, dtype=torch.float32)

        # GRPO 组内标准化: A_i = (r_i - mean(r)) / std(r)
        advantages = (returns - returns.mean()) / torch.clamp(
            returns.std(unbiased=False), min=1e-6
        )

        # 每条轨迹只有一个标量优势，复制到该轨迹的每个时间步
        expanded_advantages = [
            adv.expand(states.shape[0], 1)
            for adv, states in zip(advantages, states_list)
        ]

        return GRPOGroup(
            states=torch.cat(states_list, dim=0),
            actions=torch.cat(actions_list, dim=0),
            old_log_probs=torch.cat(old_log_probs_list, dim=0),
            old_log_probs_dist=torch.cat(old_log_probs_dist_list, dim=0),
            advantages=torch.cat(expanded_advantages, dim=0),
            returns=returns,
        )

    def update(self, group: GRPOGroup) -> None:
        """对一组轨迹做多轮 PPO-Clip + KL 惩罚更新。"""
        old_probs = group.old_log_probs_dist.exp()

        for _ in range(self.update_epochs):
            # 当前策略的完整 log 概率分布，以及实际执行动作的 log 概率
            log_probs_dist = F.log_softmax(self.pi(group.states), dim=-1)
            log_probs = log_probs_dist.gather(1, group.actions)

            # PPO-Clip: r_t = π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(log_probs - group.old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.epsilon,
                1 + self.epsilon,
            )
            surrogate = torch.min(
                ratio * group.advantages,
                clipped_ratio * group.advantages,
            )

            # KL(π_old || π_θ) 作为额外约束，防止单次更新偏离旧策略太远
            kl = (
                old_probs * (group.old_log_probs_dist - log_probs_dist)
            ).sum(dim=-1).mean()

            # 最大化 surrogate 等价于最小化 -surrogate
            loss = -surrogate.mean() + self.kl_coef * kl

            self.optimizer_pi.zero_grad()
            loss.backward()
            self.optimizer_pi.step()


def plot_training(group_ids, mean_returns, filename):
    """绘制训练曲线；matplotlib 不可用时跳过画图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过画图。")
        return

    fig, ax = plt.subplots()
    ax.plot(group_ids, mean_returns)
    ax.set_xlabel("Group")
    ax.set_ylabel("Mean return")
    ax.set_title("CartPole GRPO")
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"训练曲线已保存到 {filename}")


def main() -> None:
    torch.manual_seed(0)
    agent = Agent()
    env = gym.make("CartPole-v0")

    group_ids = []
    mean_returns = []

    for group_id in range(200):
        group = agent.sample_group(env)
        agent.update(group)

        group_ids.append(group_id)
        mean_returns.append(group.returns.mean().item())

        if (group_id + 1) % 20 == 0:
            print(
                f"Group: {group_id + 1:3d}, "
                f"mean return: {group.returns.mean().item():.2f}, "
                f"min/max: {group.returns.min().item():.1f}/"
                f"{group.returns.max().item():.1f}"
            )

    plot_training(group_ids, mean_returns, "grpo-training.pdf")
    env.close()


if __name__ == "__main__":
    main()
