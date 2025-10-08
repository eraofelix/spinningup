import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import gymnasium_robotics
import time
# MPI imports
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import proc_id, num_procs, mpi_statistics_scalar, mpi_avg, mpi_fork
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import scipy.signal
from gymnasium.spaces import Box, Discrete
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

# Constants moved from user_config.py
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__)))),'../../data')

FORCE_DATESTAMP = False


class SpatialAttention(nn.Module):
    """
    空间注意力机制，用于CNN特征图
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (B, C, H, W)
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # 应用注意力权重
        return x * attention


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    结合通道注意力和空间注意力
    """
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CNNFeatureExtractor(nn.Module):
    """
    可配置的CNN特征提取器，支持命令行参数控制
    专门为96x96x3的CarRacing图像设计
    """
    def __init__(self, input_channels=3, feature_dim=256, cnn_channels=[16, 32, 64, 128], 
                 attention_reduction=8, dropout_rate=0.1):
        super(CNNFeatureExtractor, self).__init__()
        
        # 可配置的卷积层设计
        layers = []
        prev_channels = input_channels
        
        # 第一层: 96x96 -> 48x48
        layers.extend([
            nn.Conv2d(prev_channels, cnn_channels[0], kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True)
        ])
        prev_channels = cnn_channels[0]
        
        # 第二层: 48x48 -> 24x24  
        layers.extend([
            nn.Conv2d(prev_channels, cnn_channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True)
        ])
        prev_channels = cnn_channels[1]
        
        # 第三层: 24x24 -> 12x12
        layers.extend([
            nn.Conv2d(prev_channels, cnn_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(inplace=True)
        ])
        prev_channels = cnn_channels[2]
        
        # 第四层: 12x12 -> 6x6
        layers.extend([
            nn.Conv2d(prev_channels, cnn_channels[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels[3]),
            nn.ReLU(inplace=True)
        ])
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 可配置的注意力机制
        self.attention = CBAM(cnn_channels[3], reduction=attention_reduction)
        
        # 全局平均池化 + 全连接层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels[3], feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        # x: (B, C, H, W) 或 (B, H, W, C)
        if len(x.shape) == 4 and x.shape[-1] == 3:
            # 如果是 (B, H, W, C) 格式，转换为 (B, C, H, W)
            x = x.permute(0, 3, 1, 2)
        
        # 确保输入是float类型
        if x.dtype != torch.float32:
            x = x.float()
        
        # 卷积特征提取
        features = self.conv_layers(x)
        
        # 应用注意力机制
        attended_features = self.attention(features)
        
        # 全局池化
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # 全连接层
        output = self.fc(pooled)
        
        return output


class CNNActor(nn.Module):
    """
    基于CNN的Actor网络，支持连续和离散动作空间
    """
    def __init__(self, obs_space, act_space, feature_dim=256, hidden_sizes=(128, 64), 
                 cnn_channels=[16, 32, 64, 128], attention_reduction=8, dropout_rate=0.1):
        super(CNNActor, self).__init__()
        
        self.obs_space = obs_space
        self.act_space = act_space
        
        # CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=3, 
            feature_dim=feature_dim,
            cnn_channels=cnn_channels,
            attention_reduction=attention_reduction,
            dropout_rate=dropout_rate
        )
        
        # 策略网络
        if isinstance(act_space, Box):
            # 连续动作空间
            act_dim = act_space.shape[0]
            print(f"🎯 检测到连续动作空间，维度: {act_dim}")
            
            # 为每个动作维度创建独立的log_std参数
            self.log_std = nn.Parameter(torch.zeros(act_dim))
            
            # 构建策略网络
            layers = []
            prev_size = feature_dim
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, act_dim))
            self.policy_net = nn.Sequential(*layers)
            
        elif isinstance(act_space, Discrete):
            # 离散动作空间
            act_dim = act_space.n
            print(f"🎯 检测到离散动作空间，维度: {act_dim}")
            
            layers = []
            prev_size = feature_dim
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, act_dim))
            self.policy_net = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f"Action space {act_space} not supported")
    
    def _distribution(self, obs):
        # 提取CNN特征
        features = self.cnn_extractor(obs)
        
        if isinstance(self.act_space, Box):
            # 连续动作
            mu = self.policy_net(features)
            std = torch.exp(self.log_std)
            return Normal(mu, std)
        else:
            # 离散动作
            logits = self.policy_net(features)
            return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        if isinstance(self.act_space, Box):
            return pi.log_prob(act).sum(axis=-1)
        else:
            return pi.log_prob(act)
    
    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class CNNCritic(nn.Module):
    """
    基于CNN的Critic网络
    """
    def __init__(self, obs_space, feature_dim=256, hidden_sizes=(128, 64),
                 cnn_channels=[16, 32, 64, 128], attention_reduction=8, dropout_rate=0.1):
        super(CNNCritic, self).__init__()
        
        # CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=3,
            feature_dim=feature_dim,
            cnn_channels=cnn_channels,
            attention_reduction=attention_reduction,
            dropout_rate=dropout_rate
        )
        
        # 价值网络
        layers = []
        prev_size = feature_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.value_net = nn.Sequential(*layers)
    
    def forward(self, obs):
        # 提取CNN特征
        features = self.cnn_extractor(obs)
        
        # 计算价值
        value = self.value_net(features)
        return torch.squeeze(value, -1)


class CNNActorCritic(nn.Module):
    """
    基于CNN的Actor-Critic网络，带注意力机制
    专门为CarRacing-v3等图像观测环境设计
    """
    def __init__(self, observation_space, action_space, 
                 feature_dim=256, hidden_sizes=(128, 64), cnn_channels=[16, 32, 64, 128],
                 attention_reduction=8, dropout_rate=0.1):
        super(CNNActorCritic, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # 创建共享的CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=3,
            feature_dim=feature_dim,
            cnn_channels=cnn_channels,
            attention_reduction=attention_reduction,
            dropout_rate=dropout_rate
        )
        
        # Actor网络
        self.pi = CNNActor(observation_space, action_space, feature_dim, hidden_sizes,
                          cnn_channels, attention_reduction, dropout_rate)
        
        # Critic网络  
        self.v = CNNCritic(observation_space, feature_dim, hidden_sizes,
                          cnn_channels, attention_reduction, dropout_rate)
    
    def step(self, obs):
        """
        执行一步：根据观测选择动作
        """
        with torch.no_grad():
            # 确保输入格式正确
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # 添加batch维度
            
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        
        # 返回numpy数组
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
    
    def act(self, obs):
        """
        仅获取动作，不计算其他信息
        """
        return self.step(obs)[0]


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# ============================================================================
# PPO Buffer and Agent classes
# ============================================================================

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    
    使用轨迹列表方案，无需维护指针，逻辑更简单。
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.max_size = size
        
        # 存储观测维度信息
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # 使用轨迹列表，无需指针管理
        self.trajectories = []  # 存储完整轨迹
        self.current_traj = None  # 当前正在构建的轨迹
        self.total_steps = 0  # 总步数计数器

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # 检查是否还有空间
        if self.total_steps >= self.max_size:
            return  # 缓冲区已满，忽略新数据
        
        # 如果当前轨迹为空，创建新轨迹
        if self.current_traj is None:
            self.current_traj = {
                'obs': [], 'act': [], 'rew': [], 'val': [], 'logp': []
            }
        
        # 存储数据到当前轨迹，确保数据类型一致
        self.current_traj['obs'].append(obs)
        self.current_traj['act'].append(act)
        # 确保奖励是标量
        if hasattr(rew, 'item'):
            rew = rew.item()
        self.current_traj['rew'].append(float(rew))
        # 确保价值是标量
        if hasattr(val, 'item'):
            val = val.item()
        self.current_traj['val'].append(float(val))
        # 确保对数概率是标量
        if hasattr(logp, 'item'):
            logp = logp.item()
        self.current_traj['logp'].append(float(logp))
        
        self.total_steps += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This computes advantage estimates with GAE-Lambda
        and rewards-to-go for the current trajectory.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        if self.current_traj is None or len(self.current_traj['obs']) == 0:
            return  # 没有数据需要处理
        
        # 获取当前轨迹数据，确保last_val是标量
        if hasattr(last_val, 'item'):
            last_val = last_val.item()
        last_val = float(last_val)
        
        rews = np.array(self.current_traj['rew'] + [last_val])
        vals = np.array(self.current_traj['val'] + [last_val])
        
        # 计算GAE-Lambda优势函数:                                      
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]      # 单步 δ_t = r_t + γV(s_{t+1}) - V(s_t) 
        # 假如deltas=[d0, d1, d2, d3, d4, d5] 长度为6
        # 那么adv=[a0, a1, a2, a3, a4, a5] 注意，这里adv和deltas长度相同
        # a0 = d0 + γλ * d1 + (γλ)^2 * d2 + (γλ)^3 * d3 + (γλ)^4 * d4 + (γλ)^5 * d5
        # a1 = d1 + γλ * d2 + (γλ)^2 * d3 + (γλ)^3 * d4 + (γλ)^4 * d5
        # a2 = d2 + γλ * d3 + (γλ)^2 * d4 + (γλ)^3 * d5
        # a3 = d3 + γλ * d4 + (γλ)^2 * d5
        # a4 = d4 + γλ * d5
        # a5 = d5
        # A_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k} = δ_{t} + γλ * δ_{t+1} + (γλ)^2 * δ_{t+2} + (γλ)^3 * δ_{t+3} + (γλ)^4 * δ_{t+4} + (γλ)^5 * δ_{t+5}
        adv = discount_cumsum(deltas, self.gamma * self.lam)  # 长度为6，与deltas相同
        
        # 计算回报 (rewards-to-go)
        ret = discount_cumsum(rews, self.gamma)[:-1]    # 长度为6，与rews相同
        
        # 将计算结果添加到轨迹中
        self.current_traj['adv'] = adv
        self.current_traj['ret'] = ret
        
        # 保存完整轨迹
        self.trajectories.append(self.current_traj)
        self.current_traj = None  # 重置当前轨迹

    def get(self, use_mpi=True):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets the buffer.
        """
        if not self.trajectories:
            return {}  # 没有轨迹数据
        
        # 合并所有轨迹数据
        all_obs = np.concatenate([t['obs'] for t in self.trajectories])
        all_act = np.concatenate([t['act'] for t in self.trajectories])
        all_ret = np.concatenate([t['ret'] for t in self.trajectories])
        all_adv = np.concatenate([t['adv'] for t in self.trajectories])
        all_logp = np.concatenate([t['logp'] for t in self.trajectories])
        
        all_obs = all_obs.astype(np.float32)
        
        # 归一化优势函数
        adv_mean, adv_std = mpi_statistics_scalar(all_adv)
        all_adv = (all_adv - adv_mean) / adv_std
        
        # 清空轨迹和重置计数器
        self.trajectories = []
        self.current_traj = None
        self.total_steps = 0
        
        return {
            'obs': torch.as_tensor(all_obs, dtype=torch.float32),
            'act': torch.as_tensor(all_act, dtype=torch.float32),
            'ret': torch.as_tensor(all_ret, dtype=torch.float32),
            'adv': torch.as_tensor(all_adv, dtype=torch.float32),
            'logp': torch.as_tensor(all_logp, dtype=torch.float32)
        }


class PPOAgent:
    """
    Proximal Policy Optimization (by clipping) Agent
    
    with early stopping based on approximate KL
    """
    
    def __init__(self, env_fn, actor_critic=CNNActorCritic, ac_kwargs=dict(), seed=0, 
                 steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
                 target_kl=0.05, save_freq=100, device=None, min_steps_per_proc=None):

        # Store parameters
        self.env_fn = env_fn
        self.actor_critic = actor_critic
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.save_freq = save_freq
        self.min_steps_per_proc = min_steps_per_proc
        
        # Setup device (GPU/CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"🚀 使用GPU加速: {torch.cuda.get_device_name(0)}")
                print(f"🔧 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                # 启用GPU优化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("⚡ 启用GPU优化: CUDNN benchmark")
            else:
                self.device = torch.device('cpu')
                print("💻 使用CPU训练")
        else:
            self.device = torch.device(device)
            print(f"🎯 使用指定设备: {self.device}")
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("⚡ 启用GPU优化: CUDNN benchmark")
        
        # Initialize components
        self._setup_environment()
        print("🔧 _setup_environment done")
        self._setup_agent()
        print("🔧 _setup_agent done")
        self._setup_training_components()
        print("🔧 _setup_training_components done")
    
    def _setup_environment(self):
        """Setup environment and related components"""
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        print(f"🔧 进程 {proc_id()}: 开始设置 PyTorch MPI...")
        try:
            setup_pytorch_for_mpi()
            print(f"✅ 进程 {proc_id()}: PyTorch MPI 设置完成")
        except Exception as e:
            print(f"❌ 进程 {proc_id()}: PyTorch MPI 设置失败: {e}")
            raise

        # 创建带时间戳的输出目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f'ppo_{timestamp}'
        self.output_dir = osp.join(DEFAULT_DATA_DIR, exp_name)
        
        # Set up TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.output_dir)
        
        # Save configuration to JSON file
        config_path = os.path.join(self.output_dir, 'config.json')
        os.makedirs(self.output_dir, exist_ok=True)
        import json
        
        # 只保存重要的配置参数，避免循环引用
        config_dict = {
            'env_fn': str(self.env_fn),
            'actor_critic': str(self.actor_critic),
            'ac_kwargs': self.ac_kwargs,
            'seed': self.seed,
            'steps_per_epoch': self.steps_per_epoch,
            'epochs': self.epochs,
            'gamma': self.gamma,
            'clip_ratio': self.clip_ratio,
            'pi_lr': self.pi_lr,
            'vf_lr': self.vf_lr,
            'train_pi_iters': self.train_pi_iters,
            'train_v_iters': self.train_v_iters,
            'lam': self.lam,
            'max_ep_len': self.max_ep_len,
            'target_kl': self.target_kl,
            'save_freq': self.save_freq
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # 用于存储当前 epoch 的数据
        self.epoch_metrics = {
            'ep_returns': [],
            'ep_lengths': [],
            'v_vals': [],
            'loss_pi': [],
            'loss_v': [],
            'kl': [],
            'entropy': [],
            'clip_frac': [],
            'stop_iter': [],
            'gpu_times': [],  # GPU计算时间
            'cpu_times': []   # CPU环境交互时间
        }

        # Random seed
        seed = self.seed + 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        self.env = self.env_fn()
        
        # Handle different observation spaces
        if len(self.env.observation_space.shape) == 3:
            # Image observations (H, W, C) - for CNN
            self.obs_dim = self.env.observation_space.shape
            print(f"🖼️  检测到图像观测空间: {self.obs_dim}")
        else:
            # Vector observations
            self.obs_dim = self.env.observation_space.shape[0]
            print(f"📊 检测到向量观测空间: {self.obs_dim}")
        
        self.act_dim = self.env.action_space.shape if hasattr(self.env.action_space, 'shape') else (self.env.action_space.n,)

    def _setup_agent(self):
        """Setup actor-critic agent"""
        # Create actor-critic module
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space, **self.ac_kwargs)
        
        # Move model to device (GPU/CPU)
        self.ac = self.ac.to(self.device)
        print(f"📱 模型已移动到设备: {self.device}")

        # Sync params across processes
        sync_params(self.ac)

        # Count variables (only for first process)
        if proc_id() == 0:
            var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.v])
            print(f'\nNumber of parameters: pi: {var_counts[0]}, v: {var_counts[1]}')
            print("=" * 180)
            print("Epoch    | Return    | Policy Loss | Value Loss | KL        | Entropy  | Early Stop | GPU Time | CPU Time | GPU Memory")
            print("=" * 180)

    def _setup_training_components(self):
        """Setup training components"""
        num_procs_val = num_procs()
        
        # 根据环境类型智能调整最小步数
        if self.min_steps_per_proc is not None:
            # 用户指定了最小步数
            min_steps_per_proc = self.min_steps_per_proc
        elif hasattr(self.env, 'spec') and self.env.spec.id:
            env_name = self.env.spec.id.lower()
            if 'car' in env_name or 'racing' in env_name:
                # 赛车环境需要更多步数
                min_steps_per_proc = max(self.max_ep_len * 3, 2000)
            elif 'mujoco' in env_name or 'gym' in env_name:
                # MuJoCo环境相对较短
                min_steps_per_proc = max(self.max_ep_len * 2, 1000)
            else:
                # 默认设置
                min_steps_per_proc = max(self.max_ep_len * 2, 1000)
        else:
            # 基于观测空间类型判断
            if len(self.env.observation_space.shape) == 3:
                # 图像环境（如CarRacing）需要更多步数
                min_steps_per_proc = max(self.max_ep_len * 3, 2000)
            else:
                # 向量环境
                min_steps_per_proc = max(self.max_ep_len * 2, 1000)
            
        # 计算实际步数分配
        base_steps_per_proc = int(self.steps_per_epoch / num_procs_val)
        self.local_steps_per_epoch = max(base_steps_per_proc, min_steps_per_proc)
        
        # 如果调整后总步数增加，给出警告
        total_adjusted_steps = self.local_steps_per_epoch * num_procs_val
        if total_adjusted_steps > self.steps_per_epoch:
            print(f"⚠️  步数调整: 原计划={self.steps_per_epoch}, 调整后={total_adjusted_steps}")
            print(f"🔧 进程步数分配: 总步数={total_adjusted_steps}, 进程数={num_procs_val}, 每进程步数={self.local_steps_per_epoch}")
        else:
            print(f"🔧 进程步数分配: 总步数={self.steps_per_epoch}, 进程数={num_procs_val}, 每进程步数={self.local_steps_per_epoch}")
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, self.gamma, self.lam)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def _compute_loss_pi(self, data):
        """Compute PPO policy loss"""
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Move data to device
        obs = obs.to(self.device)
        act = act.to(self.device)
        adv = adv.to(self.device)
        logp_old = logp_old.to(self.device)

        # Handle image observations: keep as images for CNN
        # CNN networks expect image format (B, C, H, W) or (B, H, W, C)
        # No flattening needed for CNN-based networks

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def _compute_loss_v(self, data):
        """Compute value function loss
        让价值函数逼近目标价值，目标价值是通过GAE (Generalized Advantage Estimation) 计算的：
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        """
        obs, ret = data['obs'], data['ret']
        
        # Move data to device
        obs = obs.to(self.device)
        ret = ret.to(self.device)

        return ((self.ac.v(obs) - ret)**2).mean()

    def _save_model(self, epoch):
        """Save model at specified epoch"""
        model_path = os.path.join(self.output_dir, f'model_epoch_{epoch}.pth')
        torch.save(self.ac.state_dict(), model_path)

    def _update(self):
        """Perform PPO update"""
        # 测量GPU训练时间
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_train_start = time.time()
        
        data = self.buf.get()

        pi_l_old, pi_info_old = self._compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_train_end = time.time()
        gpu_train_time = gpu_train_end - gpu_train_start
        
        # 记录GPU训练时间
        self.epoch_metrics['gpu_times'].append(self.epoch_metrics['gpu_times'][-1] + gpu_train_time)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        
        # 记录到 TensorBoard 指标中
        self.epoch_metrics['loss_pi'].append(pi_l_old)
        self.epoch_metrics['loss_v'].append(v_l_old)
        self.epoch_metrics['kl'].append(kl)
        self.epoch_metrics['entropy'].append(ent)
        self.epoch_metrics['clip_frac'].append(cf)
        self.epoch_metrics['stop_iter'].append(i)

    def _log_epoch_info(self, epoch, start_time):
        """Log epoch information"""
        # Print epoch info (only for first process)
        if proc_id() == 0:
            self._print_epoch_info(epoch, start_time)
    
    def _print_epoch_info(self, epoch, start_time):
        """Print epoch information"""
        # 计算平均值和统计信息
        ep_return = np.mean(self.epoch_metrics['ep_returns']) if self.epoch_metrics['ep_returns'] else 0.0
        policy_loss = np.mean(self.epoch_metrics['loss_pi']) if self.epoch_metrics['loss_pi'] else 0.0
        value_loss = np.mean(self.epoch_metrics['loss_v']) if self.epoch_metrics['loss_v'] else 0.0
        kl_div = np.mean(self.epoch_metrics['kl']) if self.epoch_metrics['kl'] else 0.0
        entropy = np.mean(self.epoch_metrics['entropy']) if self.epoch_metrics['entropy'] else 0.0
        early_stop = np.mean(self.epoch_metrics['stop_iter']) if self.epoch_metrics['stop_iter'] else 0.0
        early_stop_flag = "True" if early_stop < self.train_pi_iters - 1 else "False"
        
        # GPU性能监控和时间统计
        gpu_info = ""
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**2
            gpu_info = f" | GPU: {gpu_memory:.1f}/{gpu_max_memory:.1f}MB"
        gpu_time = self.epoch_metrics['gpu_times'][-1] if self.epoch_metrics['gpu_times'] else 0.0
        cpu_time = self.epoch_metrics['cpu_times'][-1] if self.epoch_metrics['cpu_times'] else 0.0
        total_time = gpu_time + cpu_time
        gpu_ratio = (gpu_time / total_time * 100) if total_time > 0 else 0        
        time_info = f" | GPU: {gpu_time:.2f}s({gpu_ratio:.1f}%)"
        
        # 单行打印，严格对齐
        print(f"Epoch {epoch:4d} | Return: {ep_return:5.2f} | Policy Loss: {policy_loss:5.4f} | Value Loss: {value_loss:5.4f} | KL: {kl_div:8.4f} | Entropy: {entropy:5.4f} | Early Stop: {early_stop_flag:5s}{time_info}{gpu_info}")
        
        # 记录到 TensorBoard - 基本训练指标
        self.tb_writer.add_scalar('Training/Epoch', epoch, epoch)
        self.tb_writer.add_scalar('Training/Environment_Interactions', (epoch + 1) * self.steps_per_epoch, epoch)
        self.tb_writer.add_scalar('Training/Time', time.time() - start_time, epoch)
        
        # 记录奖励和回合信息
        if self.epoch_metrics['ep_returns']:
            self.tb_writer.add_scalar('Reward/Episode_Return', np.mean(self.epoch_metrics['ep_returns']), epoch)
            if len(self.epoch_metrics['ep_returns']) > 1:
                self.tb_writer.add_scalar('Reward/Episode_Return_Std', np.std(self.epoch_metrics['ep_returns']), epoch)
            else:
                print(f"ep_returns={self.epoch_metrics['ep_returns']}")
        else:
            print(f"ep_returns={self.epoch_metrics['ep_returns']}")
        if self.epoch_metrics['ep_lengths']:
            self.tb_writer.add_scalar('Episode/Length', np.mean(self.epoch_metrics['ep_lengths']), epoch)
        
        # 记录价值、损失和策略指标
        if self.epoch_metrics['v_vals']:
            self.tb_writer.add_scalar('Values/Value_Estimates', np.mean(self.epoch_metrics['v_vals']), epoch)
        if self.epoch_metrics['loss_pi']:
            self.tb_writer.add_scalar('Loss/Policy_Loss', np.mean(self.epoch_metrics['loss_pi']), epoch)
        if self.epoch_metrics['loss_v']:
            self.tb_writer.add_scalar('Loss/Value_Loss', np.mean(self.epoch_metrics['loss_v']), epoch)
        if self.epoch_metrics['kl']:
            self.tb_writer.add_scalar('Policy/KL_Divergence', np.mean(self.epoch_metrics['kl']), epoch)
        if self.epoch_metrics['entropy']:
            self.tb_writer.add_scalar('Policy/Entropy', np.mean(self.epoch_metrics['entropy']), epoch)
        if self.epoch_metrics['clip_frac']:
            self.tb_writer.add_scalar('Policy/ClipFraction', np.mean(self.epoch_metrics['clip_frac']), epoch)
        
        # 记录训练指标和时间统计
        if self.epoch_metrics['stop_iter']:
            self.tb_writer.add_scalar('Training/StopIterations', np.mean(self.epoch_metrics['stop_iter']), epoch)
        if self.epoch_metrics['gpu_times']:
            self.tb_writer.add_scalar('Performance/GPU_Time', self.epoch_metrics['gpu_times'][-1], epoch)
        if self.epoch_metrics['cpu_times']:
            self.tb_writer.add_scalar('Performance/CPU_Time', self.epoch_metrics['cpu_times'][-1], epoch)
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            self.tb_writer.add_scalar('Performance/GPU_Memory_MB', gpu_memory, epoch)
        
        for key in self.epoch_metrics:
            self.epoch_metrics[key] = []
        
        if epoch % 10 == 0:  # Log every 10 epochs to avoid too much data
            for name, param in self.ac.named_parameters():
                self.tb_writer.add_histogram(f'Model/{name}', param, epoch)
        self.tb_writer.flush()

    def train(self):
        """Train the PPO agent"""
        # Prepare for interaction with environment
        start_time = time.time()
        o, _ = self.env.reset()
        ep_ret, ep_len = 0, 0

        # Main loop: collect experience in env and update/log each epoch
        num_debug_epochs = 1
        num_debug_steps = 0

        for epoch in range(self.epochs):
            if epoch < num_debug_epochs:
                print(f"Epoch {epoch} start")
            epoch_gpu_time = 0.0
            epoch_cpu_time = 0.0
            
            for t in range(self.local_steps_per_epoch):
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} start")
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()  # 确保GPU操作完成
                gpu_start = time.time()
                
                # Move observation to device
                obs_tensor = torch.as_tensor(o, dtype=torch.float32).to(self.device)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} obs {o.shape}")
                a, v, logp = self.ac.step(obs_tensor)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} action {a} value {v} logp {logp}")
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()  # 确保GPU操作完成
                gpu_end = time.time()
                epoch_gpu_time += (gpu_end - gpu_start)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} gpu time {gpu_end - gpu_start}")

                # CPU环境交互时间测量
                cpu_start = time.time()
                # 确保动作是正确的形状：从 (1, 3) 转换为 (3,)
                if len(a.shape) > 1 and a.shape[0] == 1:
                    action_for_env = a[0]  # 取第一个（也是唯一的）动作
                else:
                    action_for_env = a
                next_o, r, terminated, truncated, _ = self.env.step(action_for_env)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} next_o {next_o.shape} r {r} terminated {terminated} truncated {truncated}")
                cpu_end = time.time()
                epoch_cpu_time += (cpu_end - cpu_start)
                
                d = terminated or truncated  # 环境终止: 自然终止 OR 截断终止
                ep_ret += r
                ep_len += 1
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} ep_ret {ep_ret} ep_len {ep_len}")

                # save and log
                self.buf.store(o, a, r, v, logp)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} store")
                # 记录价值估计到 TensorBoard 指标中
                self.epoch_metrics['v_vals'].append(v)
                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len  # 达到最大步数限制
                terminal = d or timeout  # 轨迹结束: 自然终止 OR 超时终止
                epoch_ended = t==self.local_steps_per_epoch-1  # 当前epoch结束
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} timeout {timeout} terminal {terminal} epoch_ended {epoch_ended}")

                if terminal or epoch_ended:
                    if timeout or epoch_ended:  # 情况1: 轨迹被截断，需要引导价值
                        # 逻辑: (timeout=True) OR (epoch_ended=True) 
                        # 说明: 轨迹被强制结束，还有未来奖励，需要估计当前状态价值
                        # GPU计算时间测量
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        gpu_start = time.time()
                        obs_tensor = torch.as_tensor(o, dtype=torch.float32).to(self.device)
                        _, v, _ = self.ac.step(obs_tensor)  # 获取引导价值V(s_T)
                        if epoch < num_debug_epochs and t < num_debug_steps:
                            print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} v {v}")
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        gpu_end = time.time()
                        epoch_gpu_time += (gpu_end - gpu_start)
                    else:  # 情况2: 自然终止，不需要引导价值，HalfCheetah-v5里不存在，但其他环境可能存在
                        # 逻辑: (terminated=True) AND (truncated=False) AND (timeout=False) AND (epoch_ended=False)
                        # 说明: 任务真正结束(如智能体死亡、到达目标)，没有未来奖励
                        v = 0  # 自然终止时价值为0
                        print("自然终止")
                    self.buf.finish_path(v)  # (obs, act, rew, val, logp) -> (obs, act, ret, adv, logp, adv, ret)
                    if epoch < num_debug_epochs and t < num_debug_steps:
                        print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} finish_path")
                    # if terminal:
                        # 记录到 TensorBoard 指标中
                    self.epoch_metrics['ep_returns'].append(ep_ret)
                    self.epoch_metrics['ep_lengths'].append(ep_len)
                    o, _ = self.env.reset()
                    if epoch < num_debug_epochs and t < num_debug_steps:
                        print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} reset")
                    ep_ret, ep_len = 0, 0
            
            # 记录时间统计
            self.epoch_metrics['gpu_times'].append(epoch_gpu_time)
            self.epoch_metrics['cpu_times'].append(epoch_cpu_time)

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self._save_model(epoch)

            # Perform PPO update!
            self._update()

            # Log epoch info
            self._log_epoch_info(epoch, start_time)
        
        # Close TensorBoard writer
        self.tb_writer.close()


def ppo(env_fn, actor_critic=CNNActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.05, save_freq=100, device=None, min_steps_per_proc=None):
    agent = PPOAgent(env_fn, actor_critic, ac_kwargs, seed, steps_per_epoch, epochs, 
                    gamma, clip_ratio, pi_lr, vf_lr, train_pi_iters, train_v_iters, 
                    lam, max_ep_len, target_kl, save_freq, device, min_steps_per_proc)
    agent.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--pi_lr', type=float, default=3e-4, help='策略网络学习率')
    parser.add_argument('--vf_lr', type=float, default=1e-3, help='价值网络学习率')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--train_pi_iters', type=int, default=80, help='策略网络训练迭代次数')
    parser.add_argument('--train_v_iters', type=int, default=80, help='价值网络训练迭代次数')
    parser.add_argument('--target_kl', type=float, default=0.01, help='KL散度目标值（更保守）')
    parser.add_argument('--device', type=str, default=None, help='指定设备 (cuda/cpu/auto)')
    
    # CNN网络参数控制
    parser.add_argument('--feature_dim', type=int, default=256, help='CNN特征维度')
    parser.add_argument('--cnn_channels', type=int, nargs=4, default=[16, 32, 64, 128], 
                       help='CNN各层通道数 [conv1, conv2, conv3, conv4]')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 64], 
                       help='全连接层隐藏层大小')
    parser.add_argument('--attention_reduction', type=int, default=8, 
                       help='注意力机制reduction参数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, 
                       help='Dropout比率')
    parser.add_argument('--min_steps_per_proc', type=int, default=None,
                       help='每个进程的最小步数，用于避免轨迹截断')
    args = parser.parse_args()

    
    # 处理设备参数
    device = args.device
    if device == 'auto':
        device = None  # 让代码自动检测
    elif device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，回退到CPU")
        device = 'cpu'
    
    mpi_fork(args.cpu)  # run parallel code with mpi


    # 根据环境类型选择网络架构
    env_test = gym.make(args.env)
    if len(env_test.observation_space.shape) == 3:
        # 图像观测，使用CNN
        print("🖼️  检测到图像观测，使用CNN网络")
        actor_critic = CNNActorCritic
        ac_kwargs = dict(
            feature_dim=args.feature_dim,
            hidden_sizes=args.hidden_sizes,
            cnn_channels=args.cnn_channels,
            attention_reduction=args.attention_reduction,
            dropout_rate=args.dropout_rate
        )
    else:
        # 向量观测，使用MLP
        print("📊 检测到向量观测，使用MLP网络")
        actor_critic = MLPActorCritic
        ac_kwargs = dict(hidden_sizes=[args.hid]*args.l)
    env_test.close()

    ppo(lambda : gym.make(args.env), actor_critic=actor_critic,
        ac_kwargs=ac_kwargs, gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters, target_kl=args.target_kl,
        device=device,
        min_steps_per_proc=args.min_steps_per_proc)
