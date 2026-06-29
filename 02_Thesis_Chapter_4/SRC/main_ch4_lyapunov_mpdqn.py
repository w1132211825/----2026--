
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# 第一部分：改进型关系感知图注意力网络 (RGAT) - 对应论文 4.2.2 节
# ==============================
class RelationSpecificAttention(nn.Module):
    """
    关系特定注意力机制 (公式 4-3 至 4-5)。
    为每种关系类型分配独立的权重矩阵和注意力向量，以区分不同语义关系的强度。
    """
    def __init__(self, in_features, out_features, num_relations):
        super().__init__()
        self.num_relations = num_relations
        # 为每种关系类型初始化独立的权重矩阵 W_r 和注意力向量 a_r
        self.W_r = nn.Parameter(torch.Tensor(num_relations, in_features, out_features))
        self.a_r = nn.Parameter(torch.Tensor(num_relations, 2 * out_features))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.a_r)

    def forward(self, h, edge_index, edge_type):
        """
        参数:
            h: 节点特征矩阵 [N, in_features]
            edge_index: 边索引 [2, E]
            edge_type: 边关系类型 [E]
        返回:
            聚合后的节点特征 [N, out_features]
        """
        N = h.size(0)
        out_h = torch.zeros(N, self.W_r.size(2)).to(h.device)
        
        # 遍历每种关系类型进行独立聚合
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue
                
            src, dst = edge_index[0, mask], edge_index[1, mask]
            
            # 关系特定的特征投影 (公式 4-3 中的 W_r h)
            h_src_r = torch.matmul(h[src], self.W_r[r])
            h_dst_r = torch.matmul(h[dst], self.W_r[r])
            
            # 计算关系特定的注意力系数 e_ij (公式 4-3)
            alpha_input = torch.cat([h_src_r, h_dst_r], dim=-1)
            e_r = F.leaky_relu(torch.matmul(alpha_input, self.a_r[r]))
            
            # 归一化注意力权重 (公式 4-4)
            exp_e = torch.exp(e_r)
            sum_exp = torch.zeros(N).to(h.device)
            sum_exp.scatter_add_(0, dst, exp_e)
            alpha_r = exp_e / (sum_exp[dst] + 1e-10)
            
            # 加权聚合邻居特征 (公式 4-5)
            agg = alpha_r.unsqueeze(-1) * h_src_r
            out_h.scatter_add_(0, dst.unsqueeze(-1).expand_as(agg), agg)
            
        return F.elu(out_h)


class TimeDecayModule(nn.Module):
    """
    时间衰减模块 (公式 4-6)。
    结合三元组年龄、关系类型和上下文特征，计算自适应衰减因子。
    """
    def __init__(self, num_relations, context_dim):
        super().__init__()
        # 每种关系类型一个可学习的衰减率参数 lambda_r
        self.lambda_r = nn.Parameter(torch.ones(num_relations))
        # 上下文特征权重向量 w
        self.w = nn.Parameter(torch.randn(context_dim))
        nn.init.xavier_uniform_(self.w.unsqueeze(0))

    def forward(self, delta_t, edge_type, context_c):
        """
        参数:
            delta_t: 三元组年龄标量或向量
            edge_type: 关系类型
            context_c: 上下文特征向量 [context_dim]
        返回:
            衰减因子 beta(t)
        """
        lambda_val = self.lambda_r[edge_type]
        context_impact = torch.dot(self.w, context_c)
        # 计算衰减因子
        beta = torch.exp(-delta_t * (lambda_val - context_impact))
        return torch.clamp(beta, min=0.0, max=1.0)


class RGATModel(nn.Module):
    """
    完整的改进型 RGAT 模型 (图 4.3)。
    输入知识图谱和上下文，输出节点的重要性得分。
    """
    def __init__(self, in_features, hidden_features, num_relations, context_dim):
        super().__init__()
        self.attention_layer = RelationSpecificAttention(in_features, hidden_features, num_relations)
        self.time_decay = TimeDecayModule(num_relations, context_dim)
        # 预测层 MLP (公式 4-7)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 输出得分在 [0, 1]
        )

    def forward(self, h, edge_index, edge_type, delta_t, context_c):
        # 1. 关系特定聚合
        h_agg = self.attention_layer(h, edge_index, edge_type)
        
        # 2. 计算时间衰减因子
        beta = self.time_decay(delta_t, edge_type, context_c)
        
        # 3. 融合特征并预测得分
        logits = self.predictor(h_agg) 
        scores = logits.squeeze(-1) * beta.mean() # 应用衰减
        
        return scores


# ==============================
# 第二部分：MP-DQN 核心组件 (复用第三章架构，对应论文 4.4 节)
# ==============================
class MultiPassQActor(nn.Module):
    """多通道Q值网络（Multi-Pass Q-Actor），用于评估混合动作空间中的离散动作Q值。"""
    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=(256, 128),
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
        """前向传播：计算每个离散动作的Q值。"""
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
    """参数网络（Param-Actor），用于输出所有离散动作对应的连续动作参数（p1值）。"""
    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(256, 128), activation="relu"):
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
        
        # 功率约束参数，将在Agent初始化时注入
        self.P_total = None
        self.P_min = None

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
        if self.P_total is not None and self.P_min is not None:
            action_params = torch.sigmoid(action_params) * (self.P_total - 2 * self.P_min) + self.P_min
            
        return action_params


class Memory:
    """经验回放缓冲区（Experience Replay Buffer），用于存储和采样训练数据。"""
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


def soft_update_target_network(source_network, target_network, tau):
    """软更新目标网络参数"""
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update_target_network(source_network, target_network):
    """硬更新目标网络参数"""
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


class OrnsteinUhlenbeckActionNoise:
    """Ornstein-Uhlenbeck噪声，用于连续动作空间的探索。"""
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


class MultiPassPDQNAgent:
    """
    多通道参数化深度Q网络（Multi-Pass P-DQN）智能体。
    适用于混合动作空间（离散+连续）的强化学习问题，包含Q网络和参数网络两个子网络。
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
                 epsilon_steps=5000,
                 batch_size=64,
                 gamma=0.95,
                 P_total=0.2,  # 第四章默认总功率 200mW
                 P_min=1e-6,
                 tau_actor=0.005,
                 tau_actor_param=0.005,
                 replay_memory_size=50000,
                 learning_rate_actor=0.0005,
                 learning_rate_actor_param=0.0005,
                 initial_memory_threshold=500,
                 use_ornstein_noise=True,
                 loss_func=F.mse_loss,
                 clip_grad=10,
                 inverting_gradients=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):

        self.observation_space = observation_space
        self.action_space = action_space
        self.P_total = P_total
        self.P_min = P_min
        self.num_actions = action_space.spaces[0].n
        self.device = torch.device(device)

        # 修改：每个动作对应1个参数(p1)
        self.action_parameter_sizes = np.array([1 for _ in range(self.num_actions)])
        self.action_parameter_size = int(1 * self.num_actions)

        # 动作参数范围 [P_min, P_total - P_min]
        self.action_parameter_max_numpy = np.full(self.action_parameter_size, self.P_total - self.P_min)
        self.action_parameter_min_numpy = np.full(self.action_parameter_size, self.P_min)

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
        self.clip_grad = clip_grad
        self._step = 0
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
                                    (1 + self.action_parameter_size,))

        # 网络初始化
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
        """根据当前状态选择动作，使用epsilon-贪婪策略。"""
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
            else:
                all_action_parameters_tensor = torch.tensor(all_action_parameters).unsqueeze(0).to(self.device)
                Q_a = self.actor(state.unsqueeze(0), all_action_parameters_tensor)
                action = np.argmax(Q_a.detach().cpu().data.numpy())

            # Clip动作参数到合法范围 [P_min, P_total - P_min]
            all_action_parameters = np.clip(all_action_parameters, self.P_min, self.P_total - self.P_min)

            return action, all_action_parameters

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        """执行一步学习：存储经验并更新网络。"""
        act, all_action_parameters = action
        self._step += 1

        # 存储经验
        action_combined = np.concatenate(([act], all_action_parameters)).ravel()
        self.replay_memory.append(state, action_combined, reward, next_state, terminal)

        # 训练
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

        # 更新epsilon
        if self._step < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    self._step / self.epsilon_steps)

    def _optimize_td_loss(self):
        """优化TD损失：更新Q网络和参数网络。"""
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

        # 梯度反转：将Q网络对action_params的梯度反向传播到参数网络
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
# 第三部分：Lyapunov 增强环境 (对应论文 4.2 & 4.3 节)
# ==============================
class LyapunovOptEnv(gym.Env):
    """
    长期优化环境：引入动态语义队列与虚拟能量队列。
    状态空间扩展为公式 (4-14)，奖励函数为公式 (4-16)。
    """
    def __init__(self):
        super().__init__()
        # 基础通信参数
        self.B, self.t, self.U = 1e6, 1e-3, 1500 * 8
        self.P_total, self.P_min, self.L = 0.2, 1e-6, 10
        self.Omega_SD, self.Omega_SR, self.Omega_RD = 1e-4, 5e-4, 2e-4
        self.sigma_sq = 1e-9
        
        # Lyapunov 与队列参数
        self.P_avg = 0.15      # 长期平均功率预算 (公式 C4)
        self.Q_max = 50        # 队列最大容量
        self.V_weight = 10.0   # 权衡参数 V (公式 4-12)
        self.lam_poisson = 0.8 # 泊松到达率 lambda
        
        # 初始化队列
        self.Q_t = 0.0  # 实际数据队列
        self.Z_t = 0.0  # 虚拟能量队列
        
        # 初始化 RGAT 模型 (用于生成动态语义特征)
        self.rgat = RGATModel(in_features=16, hidden_features=32, num_relations=3, context_dim=4).to(device)
        self.rgat.eval()

        # 状态空间：11维 [信道3维, 语义特征3维, 队列2维, 常量3维] (公式 4-14)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        # 动作空间：混合动作 (M, p1)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.L),
            spaces.Box(low=self.P_min, high=self.P_total - self.P_min, shape=(self.L,), dtype=np.float32)
        ))
        
        self.max_steps = 200
        self.step_count = 0

    def _get_rgat_features(self):
        """模拟 RGAT 模块输出：计算队列中三元组的 top, avg, std 得分"""
        # 实际应用中需输入真实的图数据，此处用随机特征模拟 RGAT 的输出分布
        dummy_scores = torch.rand(int(self.Q_t) + 1).to(device) 
        if dummy_scores.numel() == 0:
            return 0.0, 0.0, 0.0
        return dummy_scores.max().item(), dummy_scores.mean().item(), dummy_scores.std().item()

    def reset(self):
        self.step_count = 0
        self.Q_t = np.random.uniform(5, 15) # 随机初始队列
        self.Z_t = 0.0
        return self._get_state()

    def _get_state(self):
        """构建增强状态向量 (公式 4-14)"""
        s_top, s_avg, s_std = self._get_rgat_features()
        return np.array([
            self.Omega_SD, self.Omega_SR, self.Omega_RD, # 信道状态
            s_top, s_avg, s_std,                         # 语义价值特征 (RGAT输出)
            self.Q_t, self.Z_t,                          # Lyapunov 队列状态
            self.P_total, self.P_avg, 1.0                # 系统常量
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        M_idx, p1_params = action
        M_idx = np.clip(M_idx, 0, self.L - 1)
        M = M_idx + 1
        p1 = np.clip(p1_params[M_idx], self.P_min, self.P_total - self.P_min)
        p2 = self.P_total - p1

        # 1. 计算瞬时语义成功概率 p_succ (复用香农公式)
        gamma_th = 2 ** (M * self.U / (self.B * self.t)) - 1
        P_out_SD = 1 - np.exp(-gamma_th * self.sigma_sq / (p1 * self.Omega_SD))
        P_out_SR = 1 - np.exp(-gamma_th * self.sigma_sq / (p1 * self.Omega_SR))
        P_out_RD = 1 - np.exp(-gamma_th * self.sigma_sq / (p2 * self.Omega_RD))
        p_succ = max(0.0, (1 - P_out_SD * (P_out_SR + (1-P_out_SR)*P_out_RD)) * 0.8)

        # 2. 更新实际队列 Q(t) (公式 4-1)
        A_t = np.random.poisson(self.lam_poisson) # 泊松到达
        # 若 p_succ 较高，则成功传输 M 个三元组
        self.Q_t = max(0, self.Q_t - M * (1 if p_succ > 0.5 else 0)) + A_t
        self.Q_t = min(self.Q_t, self.Q_max)

        # 3. 更新虚拟能量队列 Z(t) (公式 4-9)
        self.Z_t = max(0, self.Z_t + (p1 + p2) - self.P_avg)

        # 4. 计算 Lyapunov 奖励 (公式 4-16)
        # r(t) = V * p_succ + Q(t)*M - Z(t)*(p1+p2)
        reward = self.V_weight * p_succ + self.Q_t * M - self.Z_t * (p1 + p2)
        
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done, {'p_succ': p_succ, 'Q_t': self.Q_t, 'Z_t': self.Z_t}


# ==============================
# 第四部分：训练主函数 (对应论文 4.4.2 节)
# ==============================
def main():
    """主训练函数：初始化环境、Agent，执行训练循环并绘制收敛曲线。"""
    env = LyapunovOptEnv()

    # 初始化 Lyapunov 增强的 MP-DQN 智能体
    agent = MultiPassPDQNAgent(
        env.observation_space,
        env.action_space,
        batch_size=64,
        P_total=env.P_total,
        P_min=env.P_min,
        gamma=0.95,
        tau_actor=0.005,
        tau_actor_param=0.005,
        learning_rate_actor=0.0005,
        learning_rate_actor_param=0.0005,
        epsilon_steps=5000,
        replay_memory_size=50000,
        initial_memory_threshold=500,
        use_ornstein_noise=True,
        device="cpu", # 若显存充足可改为 "cuda"
        actor_kwargs={
            'hidden_layers': (256, 128),
            'activation': 'relu'
        },
        actor_param_kwargs={
            'hidden_layers': (256, 128),
            'activation': 'relu'
        }
    )

    episodes = 150  # 训练回合数
    rewards_history = []
    q_queue_history = []
    p_succ_history = []
    
    print("开始训练 Lyapunov 增强的 MP-DQN (第四章)...")
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        ep_q = []
        ep_p_succ = []
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, None, done, 1)
            
            state = next_state
            total_reward += reward
            ep_q.append(info['Q_t'])
            ep_p_succ.append(info['p_succ'])
            
        rewards_history.append(total_reward)
        q_queue_history.append(np.mean(ep_q))
        p_succ_history.append(np.mean(ep_p_succ))
        
        if (ep + 1) % 30 == 0:
            print(f"Episode {ep+1}/{episodes}: "
                  f"Avg Reward={total_reward:.2f}, "
                  f"Avg Queue Length={np.mean(ep_q):.2f}, "
                  f"Avg p_succ={np.mean(ep_p_succ):.4f}")

    # 绘制长期优化收敛曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Lyapunov Reward Convergence (公式 4-16)')
    plt.xlabel('Episode'); plt.ylabel('Cumulative Reward')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(q_queue_history)
    plt.title('Queue Stability (公式 4-2)')
    plt.xlabel('Episode'); plt.ylabel('Average Q(t)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(p_succ_history)
    plt.title('Semantic Success Probability')
    plt.xlabel('Episode'); plt.ylabel('Average p_succ')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('chapter4_lyapunov_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
