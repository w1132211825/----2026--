"""
作者：[你的姓名]
日期：2026-06-29
功能简述：基于多通道参数化深度Q网络（Multi-Pass P-DQN）的香农公式优化问题求解。
         联合优化离散动作M（数据分块数）和连续动作p1（源节点功率），
         以最大化中继通信系统的成功传输概率 p_succ。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
from collections import Counter
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# 自定义环境：基于香农公式的优化问题
# ==============================
class OptimizationEnv(gym.Env):
    """
    基于香农公式的联合离散-连续动作优化环境。
    
    状态空间：[B, t, U, sigma_sq, Omega_SD, Omega_SR, Omega_RD, P_total, L]
    动作空间：(M, p1)，其中 M ∈ {0,...,L-1} 为离散分块数，p1 ∈ [P_min, P_total-P_min] 为连续功率。
    奖励机制：p_succ * 100 + 额外奖励（成功率>0.5/0.8/0.95时给予阶梯奖励）。
    """
    def __init__(self):
        # 系统参数
        self.max_steps = 1000  # 每个episode的最大步数
        self.step_count = 0

        # 频率和带宽参数
        self.f_carrier = 2.4e9  # 载波频率 2.4 GHz (ISM频段)
        self.B = 1e6  # 带宽 1 MHz (典型WiFi带宽)

        # 时间参数
        self.t_slot = 1e-3  # 时隙长度 1ms
        self.t = 1e-3  # 单次传输时间 1ms

        # 数据传输参数
        self.packet_size = 1500  # 数据包大小 (bytes)
        self.U = self.packet_size * 8  # 转换为比特数
        self.data_rate_target = 1e6  # 目标数据率 1 Mbps

        # 信道参数 (基于实际测量)
        self.d_SD = 50.0  # S-D距离 50米
        self.d_SR = 20.0  # S-R距离 20米
        self.d_RD = 35.0  # R-D距离 35米
        self.path_loss_exp = 3.5  # 路径损耗指数 (室内环境)
        self.ref_distance = 1.0  # 参考距离 1米
        self.ref_path_loss = 30  # 参考距离路径损耗 30 dB

        # 根据距离计算信道增益
        self.Omega_SD = self._compute_channel_gain(self.d_SD)  # S-D信道增益
        self.Omega_SR = self._compute_channel_gain(self.d_SR)  # S-R信道增益
        self.Omega_RD = self._compute_channel_gain(self.d_RD)  # R-D信道增益

        # 噪声参数
        self.noise_figure = 7.0  # 接收机噪声系数 7 dB
        self.temperature = 290  # 室温 290K
        self.k_Boltzmann = 1.38e-23  # 玻尔兹曼常数
        self.sigma_sq = self._compute_noise_power()  # 噪声功率

        # 功率参数
        self.P_total = 0.1  # 总功率 100mW (20 dBm)
        self.P_min = 1e-6  # 最小功率 1μW
        self.L = 10  # 数据分块数范围 [1, 10]

        # 状态空间：更全面的系统信息
        low = np.array([
            self.B / 100, self.t / 100, self.U / 100,
            self.sigma_sq / 100, self.Omega_SD / 100, self.Omega_SR / 100, self.Omega_RD / 100,
            self.P_total / 100, 1
        ], dtype=np.float32)

        high = np.array([
            self.B * 100, self.t * 100, self.U * 100,
            self.sigma_sq * 100, self.Omega_SD * 100, self.Omega_SR * 100, self.Omega_RD * 100,
            self.P_total * 100, self.L
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 修改后的动作空间
        # 离散动作 M ∈ {0,1,...,L-1}
        # 连续动作: 每个M对应一个p1值, p2 = P_total - p1
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.L),  # M ∈ {0,1,...,L-1}
            spaces.Box(low=self.P_min, high=self.P_total - self.P_min, shape=(self.L,), dtype=np.float32)  # p1 values
        ))

        self.state = self._get_state()

    def _compute_channel_gain(self, distance):
        """根据距离计算信道增益"""
        if distance <= 0:
            return 1e-12
        path_loss_db = self.ref_path_loss + 10 * self.path_loss_exp * np.log10(distance / self.ref_distance)
        path_gain_linear = 10 ** (-path_loss_db / 10)  # 转换为线性尺度
        return path_gain_linear

    def _compute_noise_power(self):
        """计算噪声功率"""
        thermal_noise_watts = self.k_Boltzmann * self.temperature * self.B
        noise_factor = 10 ** (self.noise_figure / 10)
        total_noise_power = thermal_noise_watts * noise_factor
        return total_noise_power

    def _get_state(self):
        return np.array([
            self.B, self.t, self.U,
            self.sigma_sq, self.Omega_SD, self.Omega_SR, self.Omega_RD,
            self.P_total, self.L
        ], dtype=np.float32)

    def reset(self):
        self.state = self._get_state()
        self.step_count = 0  # 重置计数器
        return self.state

    def step(self, action):
        self.step_count += 1

        M, all_p1_parameters = action
        # 确保M在有效范围内
        M = max(0, min(M, self.L - 1))

        # 提取对应M的p1参数
        p1 = all_p1_parameters[M]

        # 计算 p2
        p2 = self.P_total - p1

        # 检查功率约束 (由于动作空间定义，p1+p2=P_total自动满足，但需要检查单个功率下限)
        if p1 < self.P_min or p2 < self.P_min:  # 这个检查其实由于动作空间定义已经满足，但保留以防万一
            reward = -10  # 温和惩罚
            done = self.step_count >= self.max_steps
            info = {
                "gamma_th": 0,
                "M": M,
                "p1": p1,
                "p2": p2,
                "P_out_SD": 1.0,
                "P_out_SR": 1.0,
                "p_succ": 0.0
            }
            return self.state, reward, done, info

        M_valid = max(M + 1, 1)  # M从0开始，实际使用时要+1
        gamma_th = 2 ** (M_valid * self.U / (self.B * self.t)) - 1

        if gamma_th <= 0:
            reward = -10
            done = self.step_count >= self.max_steps
            info = {
                "gamma_th": gamma_th,
                "M": M,
                "p1": p1,
                "p2": p2,
                "P_out_SD": 1.0,
                "P_out_SR": 1.0,
                "p_succ": 0.0
            }
            return self.state, reward, done, info

        # 计算中断概率
        P_out_SD = 1 - np.exp(-gamma_th * self.sigma_sq / (p1 * self.Omega_SD))
        P_out_SR = 1 - np.exp(-gamma_th * self.sigma_sq / (p1 * self.Omega_SR)) + \
                   np.exp(-gamma_th * self.sigma_sq / (p1 * self.Omega_SR)) * \
                   (1 - np.exp(-gamma_th * self.sigma_sq / (p2 * self.Omega_RD)))

        # 计算成功概率
        sum_term = sum([i ** (-gamma_th) for i in range(1, M_valid + 1)])
        sum_term = max(sum_term, 1e-10)
        # 注意：这里使用 log2
        min_term = min(self.B * np.log2(1 + gamma_th) * self.t / sum_term, 1)
        p_succ = (1 - P_out_SD * P_out_SR) * min_term

        # 奖励函数
        reward = p_succ * 100  # 放大奖励

        # 额外奖励：如果成功率较高
        if p_succ > 0.5:
            reward += 20
        if p_succ > 0.8:
            reward += 50
        if p_succ > 0.95:
            reward += 100

        done = self.step_count >= self.max_steps

        info = {
            "gamma_th": gamma_th,
            "M": M,
            "p1": p1,
            "p2": p2,
            "P_out_SD": P_out_SD,
            "P_out_SR": P_out_SR,
            "p_succ": p_succ
        }

        return self.state, reward, done, info


# ==============================
# 神经网络定义 (保持不变)
# ==============================
class MultiPassQActor(nn.Module):
    """
    多通道Q值网络（Multi-Pass Q-Actor）。
    
    输入状态和动作参数，输出每个离散动作对应的Q值。
    采用多通道结构，为每个离散动作单独计算Q值。
    """
    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=(100,),
                 output_layer_init_std=None, activation="relu"):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size_list = action_parameter_size_list
        self.action_parameter_size = sum(action_parameter_size_list)
        self.activation = activation

        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize

        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # 初始化权重
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        nn.init.zeros_(self.layers[-1].bias)

        self.offsets = np.concatenate(([0], np.cumsum(action_parameter_size_list)))

    def forward(self, state, action_parameters):
        """
        前向传播：计算每个离散动作的Q值。
        
        参数:
            state: 状态张量
            action_parameters: 连续动作参数张量
        
        返回:
            Q值张量，形状为 (batch_size, action_size)
        """
        negative_slope = 0.01
        Q = []
        batch_size = state.shape[0]

        # 重复输入以便处理所有动作
        x = torch.cat((state, torch.zeros_like(action_parameters)), dim=1)
        x = x.repeat(self.action_size, 1)

        for a in range(self.action_size):
            x[a * batch_size:(a + 1) * batch_size,
            self.state_size + self.offsets[a]: self.state_size + self.offsets[a + 1]] \
                = action_parameters[:, self.offsets[a]:self.offsets[a + 1]]

        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        Qall = self.layers[-1](x)

        # 提取每个动作的Q值
        for a in range(self.action_size):
            Qa = Qall[a * batch_size:(a + 1) * batch_size, a]
            if len(Qa.shape) == 1:
                Qa = Qa.unsqueeze(1)
            Q.append(Qa)
        Q = torch.cat(Q, dim=1)
        return Q


class ParamActor(nn.Module):
    """
    参数网络（Param-Actor）。
    
    输入状态，输出所有离散动作对应的连续动作参数（p1值）。
    使用sigmoid激活函数确保输出在合法范围内。
    """
    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(128, 64), activation="relu"):
        super(ParamActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        self.layers = nn.ModuleList()
        inputSize = self.state_size

        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        self.action_parameters_output_layer = nn.Linear(hidden_layers[-1] if hidden_layers else inputSize,
                                                        self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # 初始化权重
        for i in range(0, len(self.layers)):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        nn.init.xavier_normal_(self.action_parameters_output_layer.weight)
        nn.init.constant_(self.action_parameters_output_layer.bias, 0.01)
        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # 固定passthrough层
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        for layer in self.layers:
            if self.activation == "relu":
                x = F.relu(layer(x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(layer(x), negative_slope=0.01)
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)

        # 使用 sigmoid 确保输出在 [P_min, P_total - P_min] 范围内
        # 这样 p2 = P_total - p1 也会 >= P_min
        range_min = torch.full_like(action_params, 0.0)  # 相对于 P_min 的偏移
        range_max = torch.full_like(action_params, self.P_total - 2 * 0.0)  # 相对于 (P_total - P_min) 的偏移, 这里简化为 P_total
        action_params = torch.sigmoid(action_params) * (range_max - range_min) + range_min

        # 或者更直接的方式，如果 ParamActor 的 self.P_total 可访问
        # action_params = torch.sigmoid(action_params) * (self.P_total - 2 * self.P_min) + self.P_min

        # 为了简单，我们假设 ParamActor 能访问到这些值，或者在训练时通过其他方式传递
        # 这里使用一个近似值，假设 P_min 很小
        # action_params = torch.sigmoid(action_params) * self.P_total

        # 最简单的方式：在 act 方法中进行裁剪
        return action_params


# ==============================
# 内存管理 (保持不变)
# ==============================
class Memory:
    """
    经验回放缓冲区（Experience Replay Buffer）。
    
    用于存储和采样训练数据，支持循环覆盖。
    """
    def __init__(self, capacity, observation_shape, action_shape, next_actions=False):
        self.capacity = capacity
        self.next_actions = next_actions
        self.index = 0
        self.full = False

        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.terminals = np.empty((capacity,), dtype=np.uint8)

    def append(self, observation, action, reward, next_observation, terminal=False):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_observations[self.index] = next_observation
        self.terminals[self.index] = int(terminal)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size, random_machine=np.random):
        max_index = self.capacity if self.full else self.index
        indices = random_machine.choice(max_index, batch_size, replace=False)

        return (self.observations[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_observations[indices],
                self.terminals[indices])


# ==============================
# 工具函数 (保持不变)
# ==============================
def soft_update_target_network(source_network, target_network, tau):
    """软更新目标网络参数"""
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update_target_network(source_network, target_network):
    """硬更新目标网络参数"""
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


# ==============================
# PDQN Agent (修改版)
# ==============================
class MultiPassPDQNAgent:
    """
    多通道参数化深度Q网络（Multi-Pass P-DQN）智能体。
    
    适用于混合动作空间（离散+连续）的强化学习问题。
    包含Q网络（MultiPassQActor）和参数网络（ParamActor）两个子网络。
    使用经验回放和Ornstein-Uhlenbeck噪声进行探索。
    """
    NAME = "Multi-Pass P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=MultiPassQActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.02,
                 epsilon_steps=5000,  # 增加探索时间
                 batch_size=64,
                 gamma=0.95,  # 稍微增大 gamma
                 P_total=0.1,  # 从环境获取
                 P_min=1e-6,  # 新增 P_min
                 tau_actor=0.005,  # 减慢目标网络更新
                 tau_actor_param=0.005,
                 replay_memory_size=50000,  # 增大经验回放
                 learning_rate_actor=0.0005,  # 降低学习率
                 learning_rate_actor_param=0.0005,
                 initial_memory_threshold=500,  # 降低初始阈值
                 use_ornstein_noise=True,  # 启用噪声
                 loss_func=F.mse_loss,
                 clip_grad=10,
                 inverting_gradients=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):

        self.observation_space = observation_space
        self.action_space = action_space
        self.P_total = P_total
        self.P_min = P_min  # 存储 P_min
        self.num_actions = action_space.spaces[0].n
        self.device = torch.device(device)

        # 修改：每个动作对应1个参数(p1)
        self.action_parameter_sizes = np.array([1 for _ in range(self.num_actions)])
        self.action_parameter_size = int(1 * self.num_actions)

        # 动作参数范围 [P_min, P_total - P_min]
        self.action_parameter_max_numpy = np.full(self.action_parameter_size, self.P_total - self.P_min)
        self.action_parameter_min_numpy = np.full(self.action_parameter_size, self.P_min)
        self.action_parameter_range_numpy = self.action_parameter_max_numpy - self.action_parameter_min_numpy

        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)

        # epsilon贪婪策略
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.clip_grad = clip_grad
        self._step = 0
        self._episode = 0
        self.updates = 0

        # 随机数生成器
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

        # 噪声
        self.use_ornstein_noise = use_ornstein_noise
        if self.use_ornstein_noise:
            self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size,
                                                      mu=0., theta=0.15, sigma=0.2)

        # 经验回放
        self.replay_memory = Memory(replay_memory_size, observation_space.shape,
                                    (1 + self.action_parameter_size,))  # 动作维度也相应改变

        # 网络
        self.actor = actor_class(observation_space.shape[0], self.num_actions,
                                 self.action_parameter_sizes, **actor_kwargs).to(self.device)
        self.actor_target = actor_class(observation_space.shape[0], self.num_actions,
                                        self.action_parameter_sizes, **actor_kwargs).to(self.device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(observation_space.shape[0], self.num_actions,
                                             self.action_parameter_size, **actor_param_kwargs).to(self.device)
        # 注入 P_total 和 P_min 到 actor_param 网络中，以便在 forward 中使用
        self.actor_param.P_total = self.P_total
        self.actor_param.P_min = self.P_min
        self.actor_param_target = actor_param_class(observation_space.shape[0], self.num_actions,
                                                    self.action_parameter_size, **actor_param_kwargs).to(self.device)
        self.actor_param_target.P_total = self.P_total
        self.actor_param_target.P_min = self.P_min
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param)
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param

    def act(self, state):
        """
        根据当前状态选择动作。
        
        使用epsilon-贪婪策略：
        - 以概率epsilon随机选择动作（探索）
        - 以概率1-epsilon选择Q值最大的动作（利用）
        """
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            all_action_parameters = self.actor_param(state).cpu().numpy().flatten()

            # 添加噪声用于探索
            if self.use_ornstein_noise and self._step > self.initial_memory_threshold:
                noise = self.noise.sample()
                all_action_parameters += noise

            # epsilon贪婪策略
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(self.num_actions)
                # 对于随机动作，生成满足约束的随机p1
                # p1 应该在 [P_min, P_total - P_min] 范围内
                random_p1 = np.random.uniform(self.P_min, self.P_total - self.P_min)

            else:
                all_action_parameters_tensor = torch.tensor(all_action_parameters).unsqueeze(0).to(self.device)
                Q_a = self.actor(state.unsqueeze(0), all_action_parameters_tensor)
                action = np.argmax(Q_a.detach().cpu().data.numpy())

            # Clip动作参数到合法范围 [P_min, P_total - P_min]
            all_action_parameters = np.clip(all_action_parameters, self.P_min, self.P_total - self.P_min)

            return action, all_action_parameters

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        """执行一步学习：存储经验并更新网络"""
        act, all_action_parameters = action
        self._step += 1

        # 存储经验 (动作维度改变)
        action_combined = np.concatenate(([act], all_action_parameters)).ravel()
        self._add_sample(state, action_combined, reward, next_state, terminal)

        # 训练
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

        # 更新epsilon
        if self._step < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    self._step / self.epsilon_steps)

    def _add_sample(self, state, action, reward, next_state, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        """优化TD损失：更新Q网络和参数网络"""
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return

        # 采样批次数据
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze().float()

        # 更新Q网络
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()
            target = rewards + (1 - terminals) * self.gamma * Qprime

        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        loss_Q = self.loss_func(y_predicted, target)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # 更新参数网络
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad_(True)

        Q = self.actor(states, action_params)
        Q_loss = torch.mean(torch.sum(Q, 1))

        self.actor.zero_grad()
        Q_loss.backward()

        # 梯度反转
        delta_a = action_params.grad.data.clone()
        action_params_new = self.actor_param(states)

        out = -torch.mul(delta_a, action_params_new)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)
        self.actor_param_optimiser.step()

        # 软更新目标网络
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)


# ==============================
# Ornstein-Uhlenbeck噪声 (保持不变)
# ==============================
class OrnsteinUhlenbeckActionNoise:
    """
    Ornstein-Uhlenbeck噪声，用于连续动作空间的探索。
    
    这是一种均值回归的随机过程，适合在连续动作空间中提供时间相关的探索噪声。
    """
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2, random_machine=np.random):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.random_machine = random_machine
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.random_machine.randn(len(x))
        self.state = x + dx
        return self.state


# ==============================
# 训练脚本 (修改版)
# ==============================
def main():
    """主训练函数"""
    env = OptimizationEnv()

    agent = MultiPassPDQNAgent(
        env.observation_space,
        env.action_space,
        batch_size=64,
        P_total=env.P_total,
        P_min=env.P_min,  # 传递 P_min
        gamma=0.95,
        tau_actor=0.005,
        tau_actor_param=0.005,
        learning_rate_actor=0.0005,
        learning_rate_actor_param=0.0005,
        epsilon_steps=5000,
        replay_memory_size=50000,
        initial_memory_threshold=500,
        use_ornstein_noise=True,
        device="cpu",
        actor_kwargs={
            'hidden_layers': (256, 128),  # 增加网络容量
            'activation': 'relu'
        },
        actor_param_kwargs={
            'hidden_layers': (256, 128),
            'activation': 'relu'
        }
    )

    episodes = 200  # 增加训练轮数
    rewards = []
    avg_p_succ_history = []
    max_p_succ_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        episode_p_succ = []

        while not done and step_count < env.max_steps:
            action = agent.act(state)
            act, act_param = action
            next_state, reward, done, info = env.step((act, act_param))
            agent.step(state, (act, act_param), reward, next_state, None, done, 1)
            state = next_state
            total_reward += reward
            step_count += 1

            if 'p_succ' in info and info['p_succ'] is not None:
                episode_p_succ.append(info['p_succ'])

        avg_p_succ = np.mean(episode_p_succ) if episode_p_succ else 0
        max_p_succ = np.max(episode_p_succ) if episode_p_succ else 0
        avg_p_succ_history.append(avg_p_succ)
        max_p_succ_history.append(max_p_succ)

        print(f"Episode {ep} Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
        print(f"  Steps: {step_count}, Avg p_succ: {avg_p_succ:.4f}, Max p_succ: {max_p_succ:.4f}")
        if episode_p_succ:
            final_info = info  # 保存最后一步的info用于打印
            print(
                f"  Final M: {final_info.get('M', 'N/A')}, p1: {final_info.get('p1', 'N/A'):.6f}, p2: {final_info.get('p2', 'N/A'):.6f}")
            print(f"  Final p_succ: {final_info.get('p_succ', 'N/A'):.4f}")
        rewards.append(total_reward)

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(avg_p_succ_history, label='Average p_succ')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Average Success Rate per Episode')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(max_p_succ_history, label='Max p_succ', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Max Success Rate per Episode')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
