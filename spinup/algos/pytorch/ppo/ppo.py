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
from gymnasium.wrappers import FrameStackObservation as FrameStack

DEFAULT_DATA_DIR = "/root/tf-logs" if osp.exists("/root/tf-logs") else osp.join(osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__)))),'../../data')
FORCE_DATESTAMP = False

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

def atanh(x, eps=1e-6):
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

class TanhNormal:
    """
    Tanh-squashed Gaussian with correct log_prob (includes tanh Jacobian).
    pi(z) = N(mu, std); a = tanh(z)
    log_prob(a) = log N(z | mu, std) - sum log(1 - tanh(z)^2)
                where z = atanh(a)
    Supports sampling by reparameterization.
    """
    def __init__(self, mu, log_std, eps=1e-6):
        self.mu = mu
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.base_dist = Normal(mu, self.std)
        self.eps = eps

    def sample(self):
        z = self.base_dist.rsample()
        a = torch.tanh(z)
        return a

    def rsample_with_pre_tanh(self):
        z = self.base_dist.rsample()
        a = torch.tanh(z)
        return a, z

    def log_prob(self, a):
        # Inverse transform
        z = atanh(a, self.eps)
        log_prob_gauss = self.base_dist.log_prob(z)  # shape: (..., act_dim)
        # log |det J| for tanh: sum log(1 - tanh(z)^2) = sum log(1 - a^2)
        log_det = torch.log(1 - a.pow(2) + self.eps)
        return (log_prob_gauss - log_det).sum(dim=-1)

    def entropy(self, num_samples=1):
        # 使用未squash的Normal熵作为代理，保持与log_prob计算的一致性
        # H = 0.5 * Σ(1 + log(2πσ²)) = 0.5 * Σ(1 + log(2π) + 2*log(σ))
        # 简化为: H = 0.5 * Σ(1 + log(2π) + 2*log_std)
        log_2pi = np.log(2 * np.pi)
        entropy = 0.5 * (1 + log_2pi + 2 * self.log_std).sum(dim=-1)
        return entropy

class SimpleSharedCNN(nn.Module):
    """
    Lightweight CNN for 96x96 inputs with configurable channels. No BN/Dropout.
    Outputs a feature vector of size feature_dim.
    Supports FrameStack: RGB(3) -> RGB+Stack(12) or Grayscale(1) -> Grayscale+Stack(4)
    """
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        确定性重排到 (B, C, H, W) 并无条件归一化到 [0,1]
        允许以下输入形状：
          - (B, S, H, W, C)   # batched FrameStack Gymnasium
          - (S, H, W, C)      # single FrameStack
          - (B, H, W, C)      # batched NHWC
          - (B, C, H, W)      # batched NCHW
          - (C, H, W)         # single NCHW
          - (H, W, C)         # single NHWC
        """
        orig_shape = x.shape
        if x.dim() == 5:
            # (B, S, H, W, C) -> (B, S*C, H, W)
            B, S, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).reshape(B, S * C, H, W)
        elif x.dim() == 4:
            # 可能是 (B,H,W,C) / (B,C,H,W) / (S,H,W,C)
            if x.shape[-1] in (1, 3):  # NHWC-like
                if x.shape[0] > 8 and x.shape[1] == x.shape[2] and x.shape[-1] in (1, 3):
                    # (B,H,W,C)
                    x = x.permute(0, 3, 1, 2)  # -> (B,C,H,W)
                else:
                    # (S,H,W,C) 视作单样本堆叠
                    S, H, W, C = x.shape
                    x = x.permute(0, 3, 1, 2).reshape(1, S * C, H, W)  # -> (1, S*C, H, W)
            else:
                # 假定已经是 (B,C,H,W)
                pass
        elif x.dim() == 3:
            # (C,H,W) / (H,W,C)
            if x.shape[0] in (1, 3, 4, 12):
                x = x.unsqueeze(0)  # -> (1,C,H,W)
            elif x.shape[-1] in (1, 3):
                H, W, C = x.shape
                x = x.permute(2, 0, 1).unsqueeze(0)  # -> (1,C,H,W)
            else:
                raise ValueError(f"无法判定输入通道排列，shape={orig_shape}")
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}, shape={orig_shape}")

        # 校验通道
        expected_c = self.conv[0].in_channels
        if x.shape[1] != expected_c:
            raise ValueError(f"通道数不匹配: 输入 {x.shape[1]}, 期望 {expected_c}, 原始形状 {orig_shape}")

        # 无条件 /255 归一化
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0

        feats = self.conv(x)
        feats = self.head(feats)
        return feats

class ActorHead(nn.Module):
    """
    Gaussian policy head with Tanh squashing to [-1,1].
    """
    def __init__(self, feature_dim, act_dim, hidden_sizes=(256,128), init_log_std=-1.0):
        super().__init__()
        mlp = []
        in_dim = feature_dim
        for h in hidden_sizes:
            mlp += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        self.mlp = nn.Sequential(*mlp)
        self.mu_layer = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def forward(self, feats):
        h = self.mlp(feats)
        mu = self.mu_layer(h)
        log_std = self.log_std.clamp(-4, 1)
        return mu, log_std

class CriticHead(nn.Module):
    def __init__(self, feature_dim, hidden_sizes=(256,128)):
        super().__init__()
        mlp = []
        in_dim = feature_dim
        for h in hidden_sizes:
            mlp += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        mlp += [nn.Linear(in_dim, 1)]
        self.v = nn.Sequential(*mlp)

    def forward(self, feats):
        return self.v(feats).squeeze(-1)

class CNNActorCriticShared(nn.Module):
    """
    Shared CNN + separate heads. Supports:
    - Continuous Box actions with Tanh-squashed Gaussian
    - Discrete actions (optional path if needed)
    """
    def __init__(self, observation_space, action_space,
                 feature_dim=256, actor_hidden=(256,128), critic_hidden=(256,128),
                 car_racing_mode=True, use_framestack=True):
        super().__init__()
        self.obs_space = observation_space
        self.act_space = action_space
        self.is_box = isinstance(action_space, Box)
        self.is_discrete = isinstance(action_space, Discrete)
        assert self.is_box or self.is_discrete, "Unsupported action space"

        # 根据观测空间确定输入通道数
        if len(observation_space.shape) == 4:  # FrameStack后的形状 (stack_size, H, W, C)
            # FrameStack后的观测空间
            stack_size, h, w, c = observation_space.shape
            in_channels = stack_size * c  # 总通道数 = 堆叠数 × 单帧通道数
        elif len(observation_space.shape) == 3:  # 单帧图像观测 (H, W, C)
            # 单帧图像观测
            if observation_space.shape[-1] == 3:  # RGB
                in_channels = 3
            else:  # 灰度
                in_channels = 1
        else:
            in_channels = 3  # 默认RGB
        
        print(f"🔧 CNN输入通道数: {in_channels} (FrameStack: {use_framestack})")
        
        # Shared CNN
        self.encoder = SimpleSharedCNN(in_channels=in_channels, feature_dim=feature_dim)

        if self.is_box:
            act_dim = action_space.shape[0]
            self.pi = ActorHead(feature_dim, act_dim, hidden_sizes=actor_hidden)
            self.v = CriticHead(feature_dim, hidden_sizes=critic_hidden)
        else:
            act_dim = action_space.n
            self.policy_logits = nn.Sequential(
                nn.Linear(feature_dim, actor_hidden[0]), nn.ReLU(inplace=True),
                nn.Linear(actor_hidden[0], act_dim)
            )
            self.v = CriticHead(feature_dim, hidden_sizes=critic_hidden)

        # CarRacing 专用动作映射开关
        self.car_racing_mode = car_racing_mode and self.is_box and action_space.shape[0] == 3

    # ---------- Policy distribution ----------

    def _pi_dist(self, obs):
        feats = self.encoder(obs)
        if self.is_box:
            mu, log_std = self.pi(feats)
            return TanhNormal(mu, log_std), feats
        else:
            logits = self.policy_logits(feats)
            from torch.distributions import Categorical
            return Categorical(logits=logits), feats

    def _log_prob_from_dist(self, pi, act):
        if self.is_box:
            # act expected in [-1,1] (tanh range)
            return pi.log_prob(act)
        else:
            return pi.log_prob(act)

    # ---------- CarRacing action mapping ----------

    @staticmethod
    def _map_to_carracing(a_tanh, prev_action=None, steering_smooth=0.1):
        """
        Input a_tanh in [-1,1]^3.
        Output with action hygiene:
          steer in [-1,1] (with smoothing)
          gas   in [0,1]
          brake in [0,1] (with suppression)
        """
        steer = a_tanh[..., 0]
        gas   = (a_tanh[..., 1] + 1) * 0.5  # [-1,1] -> [0,1]
        brake = (a_tanh[..., 2] + 1) * 0.5  # [-1,1] -> [0,1]
        
        # 刹车抑制：当油门>0.1时，抑制刹车
        brake_suppressed = torch.where(gas > 0.1, brake * 0.1, brake)
        
        # 转向平滑：如果有前一个动作，进行平滑
        if prev_action is not None:
            prev_steer = prev_action[..., 0]
            steer_smoothed = (1 - steering_smooth) * steer + steering_smooth * prev_steer
        else:
            steer_smoothed = steer
            
        return torch.stack([steer_smoothed, gas, brake_suppressed], dim=-1)

    @staticmethod
    def _log_prob_carracing_from_tanh(pi, a_env):
        """
        If you want exact log_prob for env action space (after affine mapping),
        you can transform back:
          a_tanh[:,1:3] = 2*gas/brake - 1
        Then add log|det J| of affine (constant): for each dim in [1,2],
          scale s=2.0 -> log|s|=log 2. Sum across dims. This is constant wrt params.
        In practice, PPO with constant offset in log_prob is okay to ignore for ratio,
        but for completeness we include it.
        """
        eps = 1e-6
        steer = a_env[..., 0].clamp(-1.0, 1.0)
        gas   = a_env[..., 1].clamp(0.0, 1.0)
        brake = a_env[..., 2].clamp(0.0, 1.0)
        a_tanh = torch.stack([steer, gas*2-1, brake*2-1], dim=-1).clamp(-1+eps, 1-eps)
        base_logp = pi.log_prob(a_tanh)  # includes tanh Jacobian
        # affine mapping for gas/brake: a_env = (a_tanh+1)/2 -> da_env/da_tanh = 1/2
        # log|det J_affine| = sum log(1/2) over dims 1 and 2 = 2 * log(1/2) = -2*log 2
        # For log_prob of a_env, need to add log|det d a_tanh / d a_env| = -log|det J_affine|
        # Here we want p(a_env) = p(a_tanh) * |det(d a_tanh / d a_env)|
        # det(d a_tanh / d a_env) = 2 * 2 = 4 -> log 4
        log_det = torch.log(torch.tensor(4.0, device=a_env.device))
        return base_logp + log_det

    # ---------- Public API ----------

    def step(self, obs, return_env_action=True):
        """
        obs: (B,3,H,W) or (3,H,W). Returns:
          - action np.array (tanh space for training consistency)
          - value np.array
          - logp np.array (for the exact action fed back into policy gradient)
        If return_env_action=True and car_racing_mode, returns mapped env action.
        """
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            pi, feats = self._pi_dist(obs)
            if self.is_box:
                a_tanh = pi.sample()  # in [-1,1]
                v = self.v(feats)
                logp = pi.log_prob(a_tanh)  # 统一使用tanh空间的log_prob
                
                if self.car_racing_mode and return_env_action:
                    a_env = self._map_to_carracing(a_tanh)
                    # 返回环境动作用于环境交互，但logp始终是tanh空间
                    return a_env.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()
                else:
                    return a_tanh.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()
            else:
                a = pi.sample()
                v = self.v(feats)
                logp = pi.log_prob(a)
                return a.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()

    def act(self, obs, return_env_action=True):
        return self.step(obs, return_env_action=return_env_action)[0]

    # ---------- Training-time forward ----------

    def pi_and_logp(self, obs, act, assume_env_action=True):
        """
        obs: tensor (B,3,H,W)
        act: tensor (B, act_dim)
        assume_env_action:
          - True: act 是 CarRacing 的 env 动作（[-1,1], [0,1], [0,1]），将其映射回 tanh 空间计算 logp
          - False: act 已经是 tanh 空间 [-1,1]，直接用
        Returns: pi_dist, logp
        """
        pi, _ = self._pi_dist(obs)
        if self.is_box:
            if self.car_racing_mode and assume_env_action:
                logp = self._log_prob_carracing_from_tanh(pi, act)
            else:
                logp = pi.log_prob(act)
            return pi, logp
        else:
            from torch.distributions import Categorical
            logp = pi.log_prob(act)
            return pi, logp

    def value(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        feats = self.encoder(obs)
        return self.v(feats)
    
    def _pi_dist_from_params(self, mu, log_std):
        """从参数创建策略分布"""
        if self.is_box:
            return TanhNormal(mu, log_std)
        else:
            from torch.distributions import Categorical
            return Categorical(logits=mu)

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.9, adv_clip_range=3.0):
        self.gamma, self.lam = gamma, lam
        self.max_size = size
        self.adv_clip_range = adv_clip_range  # 优势函数裁剪范围
        
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
        self.current_traj['deltas'] = deltas  # 存储delta用于统计
        
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
        all_deltas = np.concatenate([t['deltas'] for t in self.trajectories])
        # all_obs = all_obs.astype(np.float32)
        
        # 统计GAE数值（规范化前）
        self._print_gae_statistics(all_adv, all_ret, all_deltas)
        
        # 对优势函数进行轻裁剪（winsorize），抑制长尾样本
        all_adv = np.clip(all_adv, -self.adv_clip_range, self.adv_clip_range)
        from spinup.utils.mpi_tools import proc_id
        if proc_id() == 0:
            print(f"  优势函数裁剪后: 均值={all_adv.mean():.6f}, 标准差={all_adv.std():.6f}")
        
        # 归一化优势函数
        adv_mean, adv_std = mpi_statistics_scalar(all_adv)
        all_adv = (all_adv - adv_mean) / adv_std
        
        # 如果标准差仍然太小，应用增强系数
        if adv_std < 0.95:
            boost_factor = 1.2
            all_adv = all_adv * boost_factor
            from spinup.utils.mpi_tools import proc_id
            if proc_id() == 0:
                print(f"  优势函数增强: std={adv_std:.3f} < 0.95, 应用增强系数 {boost_factor}")
                print(f"  增强后优势函数: 均值={all_adv.mean():.6f}, 标准差={all_adv.std():.6f}")
        
        # 验证规范化后的优势函数
        self._print_normalized_adv_statistics(all_adv)
        
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
    
    def _print_gae_statistics(self, adv, ret, deltas):
        """打印GAE统计信息（仅在进程0打印）"""
        from spinup.utils.mpi_tools import proc_id
        if proc_id() != 0:
            return
            
        print(f"\n📊 GAE统计信息:")
        print(f"  优势函数 (裁剪前):")
        print(f"    均值: {adv.mean():.6f}")
        print(f"    标准差: {adv.std():.6f}")
        print(f"    最小值: {adv.min():.6f}")
        print(f"    最大值: {adv.max():.6f}")
        
        # 检查是否有极端值
        extreme_positive = np.sum(adv > self.adv_clip_range)
        extreme_negative = np.sum(adv < -self.adv_clip_range)
        if extreme_positive > 0 or extreme_negative > 0:
            print(f"    极端值: {extreme_positive} 个 > {self.adv_clip_range}, {extreme_negative} 个 < -{self.adv_clip_range}")
            print(f"    💡 将进行裁剪以抑制长尾样本影响")
        
        print(f"  回报统计:")
        print(f"    均值: {ret.mean():.6f}")
        print(f"    标准差: {ret.std():.6f}")
        print(f"    最小值: {ret.min():.6f}")
        print(f"    最大值: {ret.max():.6f}")
        print(f"    百分位数 - P5: {np.percentile(ret, 5):.6f}, P50: {np.percentile(ret, 50):.6f}, P95: {np.percentile(ret, 95):.6f}")
        
        print(f"  Delta统计 (δ = r + γV(s') - V(s)):")
        print(f"    均值: {deltas.mean():.6f}")
        print(f"    标准差: {deltas.std():.6f}")
        print(f"    最小值: {deltas.min():.6f}")
        print(f"    最大值: {deltas.max():.6f}")
        print(f"    百分位数 - P5: {np.percentile(deltas, 5):.6f}, P50: {np.percentile(deltas, 50):.6f}, P95: {np.percentile(deltas, 95):.6f}")
        
        # 检查问题
        if abs(adv.mean()) < 1e-6 and adv.std() < 1e-6:
            print(f"  ⚠️  优势函数几乎为0，PPO梯度信号很弱！")
        if ret.std() < 1e-6:
            print(f"  ⚠️  回报几乎恒定，环境可能有问题！")
        if ret.mean() < -100:
            print(f"  ⚠️  回报过低，可能需要调整奖励设计！")
        if abs(deltas.mean()) < 1e-6 and deltas.std() < 1e-6:
            print(f"  ⚠️  Delta几乎为0，价值函数可能没有学习！")
        if deltas.std() > 100:
            print(f"  ⚠️  Delta标准差过大，可能存在梯度爆炸！")
    
    def _print_normalized_adv_statistics(self, adv_normalized):
        """打印规范化后的优势函数统计（仅在进程0打印）"""
        from spinup.utils.mpi_tools import proc_id
        if proc_id() != 0:
            return
            
        print(f"  优势函数 (规范化后):")
        print(f"    均值: {adv_normalized.mean():.6f} (应接近0)")
        print(f"    标准差: {adv_normalized.std():.6f} (应接近1)")
        print(f"    最小值: {adv_normalized.min():.6f}")
        print(f"    最大值: {adv_normalized.max():.6f}")
        
        # 验证规范化效果
        if abs(adv_normalized.mean()) > 0.1:
            print(f"  ⚠️  规范化后均值偏离0太多: {adv_normalized.mean():.6f}")
        if abs(adv_normalized.std() - 1.0) > 0.1:
            print(f"  ⚠️  规范化后标准差偏离1太多: {adv_normalized.std():.6f}")

class PPOAgent:
    def __init__(self, env_fn, actor_critic, ac_kwargs=dict(), seed=0, 
                 steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.1, pi_lr=3e-4,
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
        
        # 为CarRacing环境添加FrameStack
        if hasattr(self.env, 'spec') and self.env.spec and 'CarRacing' in self.env.spec.id:
            print("🏎️  检测到CarRacing环境，添加FrameStack(4)包装器")
            self.env = FrameStack(self.env, stack_size=4)
            print(f"📊 FrameStack后观测空间: {self.env.observation_space}")
        
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

        self.local_steps_per_epoch = self.steps_per_epoch
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, self.gamma, self.lam, adv_clip_range=3.0)

        # Set up optimizers for policy and value function
        # 策略优化器优化encoder+pi，确保encoder参与策略学习
        self.pi_optimizer = Adam(list(self.ac.encoder.parameters()) + list(self.ac.pi.parameters()), lr=self.pi_lr)
        # 价值优化器只优化value头，避免重复优化encoder
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        self.minibatch_size = 2048  # 如果总样本少于 2048，可设为 512 或 1024
        self.policy_epochs = 3
        self.value_epochs = 4

        self.kl_history = []
        self.cf_history = []

    def _compute_loss_pi(self, data):
        """Compute PPO policy loss - 优化encoder+pi，确保encoder参与策略学习"""
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Move data to device
        obs = obs.to(self.device)
        act = act.to(self.device)
        adv = adv.to(self.device)
        logp_old = logp_old.to(self.device)

        # 策略损失 - encoder参与梯度计算
        feats = self.ac.encoder(obs)  # 移除torch.no_grad()，让encoder参与梯度计算
        
        # 计算pi头的损失
        mu, log_std = self.ac.pi(feats)
        pi = self.ac._pi_dist_from_params(mu, log_std)
        logp = pi.log_prob(act)
        
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
        obs, ret = data['obs'], data['ret']
        obs = obs.to(self.device); ret = ret.to(self.device)
        feats = self.ac.encoder(obs).detach()
        v = self.ac.v(feats)
        return F.smooth_l1_loss(v, ret)

    def _save_model(self, epoch):
        """Save model at specified epoch"""
        model_path = os.path.join(self.output_dir, f'model_epoch_{epoch}.pth')
        torch.save(self.ac.state_dict(), model_path)

    def _iterate_minibatches(self, data_dict, batch_size, shuffle=True):
        """
        将整批数据切分为小批次生成器。data_dict 的每个 value 是 tensor，shape[0]==N。
        """
        N = data_dict['obs'].shape[0]
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            mb_idx = idx[start:end]
            yield {k: v[mb_idx] for k, v in data_dict.items()}

    def _update(self):
        """
        小批次 PPO 更新 + 严格 KL 早停 + pi_lr 自适应
        需要在 _setup_training_components 里设置：
        self.minibatch_size = 2048 (或更小，如总样本<2048则取512)
        self.policy_epochs = 3
        self.value_epochs = 4
        并在 __init__ 中初始化：
        self.kl_history = []
        self.cf_history = []
        """
        # 准备数据
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_train_start = time.time()

        data = self.buf.get()
        # 预先把数据放到设备上，便于切小批时直接索引
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(self.device)

        # 计算整批的 old loss（仅日志用途）
        with torch.no_grad():
            pi_l_old, pi_info_old = self._compute_loss_pi(data)
            v_l_old = self._compute_loss_v(data)
            pi_l_old = pi_l_old.item()
            v_l_old = v_l_old.item()

        # 策略小批多 epoch
        kl_list_epoch = []
        cf_list_epoch = []
        for pe in range(getattr(self, 'policy_epochs', 3)):
            # 每个 policy epoch 遍历所有小批
            for mb in self._iterate_minibatches(data, getattr(self, 'minibatch_size', 2048), shuffle=True):
                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = self._compute_loss_pi(mb)

                # KL 取 MPI 平均
                kl = mpi_avg(pi_info['kl'])
                cf = mpi_avg(pi_info['cf'])
                # 严格 KL 早停
                if kl > self.target_kl:
                    # 不 step，直接停止本轮剩余小批
                    break

                loss_pi.backward()
                mpi_avg_grads(self.ac.encoder)
                mpi_avg_grads(self.ac.pi)
                self.pi_optimizer.step()

                kl_list_epoch.append(kl)
                cf_list_epoch.append(cf)
            # 若已超过 KL，终止后续 policy epochs
            if len(kl_list_epoch) > 0 and np.mean(kl_list_epoch) > self.target_kl:
                break

        # 价值小批多 epoch（SmoothL1Loss 内部在 _compute_loss_v 实现）
        for ve in range(getattr(self, 'value_epochs', 4)):
            for mb in self._iterate_minibatches(data, getattr(self, 'minibatch_size', 2048), shuffle=True):
                self.vf_optimizer.zero_grad()
                loss_v = self._compute_loss_v(mb)
                loss_v.backward()
                mpi_avg_grads(self.ac.v)
                self.vf_optimizer.step()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_train_time = time.time() - gpu_train_start
        self.epoch_metrics['gpu_times'].append(self.epoch_metrics['gpu_times'][-1] + gpu_train_time)

        # 记录本 epoch 的 KL/CF
        mean_kl = float(np.mean(kl_list_epoch)) if kl_list_epoch else 0.0
        mean_cf = float(np.mean(cf_list_epoch)) if cf_list_epoch else 0.0
        self.kl_history.append(mean_kl)
        self.cf_history.append(mean_cf)
        if len(self.kl_history) > 20:
            self.kl_history = self.kl_history[-20:]
            self.cf_history = self.cf_history[-20:]

        # 自适应 pi_lr（每 3 个 epoch 判一次）
        if len(self.kl_history) >= 3:
            recent_kl = np.mean(self.kl_history[-3:])
            recent_cf = np.mean(self.cf_history[-3:])
            new_lr = None
            if (recent_kl < 0.5 * self.target_kl) and (recent_cf < 0.1):
                new_lr = min(self.pi_lr * 1.5, 2e-4)
            elif (recent_kl > 2.0 * self.target_kl) or (recent_cf > 0.4):
                new_lr = max(self.pi_lr * 0.5, 1e-5)
            if new_lr is not None and abs(new_lr - self.pi_lr) / self.pi_lr > 0.01:
                for g in self.pi_optimizer.param_groups:
                    g['lr'] = new_lr
                self.pi_lr = new_lr  # 记录当前 lr

        # 写入指标（注意：这里用 old 的 pi_l_old/v_l_old 作为 epoch 级损失参考）
        ent_log = pi_info_old['ent'] if isinstance(pi_info_old, dict) and 'ent' in pi_info_old else 0.0
        self.epoch_metrics['loss_pi'].append(pi_l_old)
        self.epoch_metrics['loss_v'].append(v_l_old)
        self.epoch_metrics['kl'].append(mean_kl)
        self.epoch_metrics['entropy'].append(ent_log)
        self.epoch_metrics['clip_frac'].append(mean_cf)
        self.epoch_metrics['stop_iter'].append(0)  # 不再用迭代计数作为早停标志

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
        
        gpu_time = self.epoch_metrics['gpu_times'][-1] if self.epoch_metrics['gpu_times'] else 0.0
        cpu_time = self.epoch_metrics['cpu_times'][-1] if self.epoch_metrics['cpu_times'] else 0.0
        total_time = gpu_time + cpu_time
        gpu_ratio = (gpu_time / total_time * 100) if total_time > 0 else 0        
        time_info = f" | GPU: {gpu_time:.2f}s({gpu_ratio:.1f}%)"
        
        # 单行打印，严格对齐
        print(f"Epoch {epoch:4d} | Return: {ep_return:5.2f} | Policy Loss: {policy_loss:5.4f} | Value Loss: {value_loss:5.4f} | KL: {kl_div:8.4f} | Entropy: {entropy:5.4f} | Early Stop: {early_stop_flag:5s}{time_info}")
        
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
        prev_action = None  # 用于转向平滑

        # Main loop: collect experience in env and update/log each epoch
        num_debug_epochs = 3
        num_debug_steps = 3
        reward_scale = getattr(self, 'reward_scale', 3.0)
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
                
                # Move observation to device - 确保观测是uint8格式存储到buffer
                # 对于CNN，我们让CNN自己处理归一化，这里保持原始格式
                obs_tensor = torch.as_tensor(o, dtype=torch.float32).to(self.device)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} obs {o.shape}")
                
                # 获取tanh空间的动作和logp（用于训练）
                a_tanh, v, logp = self.ac.step(obs_tensor, return_env_action=False)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} tanh action {a_tanh} value {v} logp {logp}")
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()  # 确保GPU操作完成
                gpu_end = time.time()
                epoch_gpu_time += (gpu_end - gpu_start)
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} gpu time {gpu_end - gpu_start}")

                # CPU环境交互时间测量
                cpu_start = time.time()
                # 将tanh动作转换为环境动作用于环境交互，应用动作卫生处理
                if hasattr(self.ac, 'car_racing_mode') and self.ac.car_racing_mode:
                    # 使用CarRacing动作映射，包含刹车抑制和转向平滑
                    a_tanh_tensor = torch.FloatTensor(a_tanh)
                    prev_action_tensor = torch.FloatTensor(prev_action) if prev_action is not None else None
                    action_for_env = self.ac._map_to_carracing(a_tanh_tensor, prev_action_tensor).cpu().numpy()
                else:
                    # 直接使用tanh动作
                    action_for_env = a_tanh
                
                # 确保动作是正确的形状
                if len(action_for_env.shape) > 1 and action_for_env.shape[0] == 1:
                    action_for_env = action_for_env[0]
                
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} env action {action_for_env}")
                
                next_o, r, terminated, truncated, _ = self.env.step(action_for_env)
                ep_ret += r
                r_scaled = r / reward_scale
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} next_o {next_o.shape} r {r} terminated {terminated} truncated {truncated}")
                cpu_end = time.time()
                epoch_cpu_time += (cpu_end - cpu_start)
                
                # 更新前一个动作用于转向平滑
                prev_action = action_for_env.copy()
                
                d = terminated or truncated  # 环境终止: 自然终止 OR 截断终止
                ep_ret += r
                ep_len += 1
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} ep_ret {ep_ret} ep_len {ep_len}")

                # save and log - 存储tanh空间的动作用于训练一致性
                # 确保存储的观测是uint8格式，避免重复归一化
                if o.dtype != np.uint8:
                    o = o.astype(np.uint8)
                self.buf.store(o, a_tanh, r_scaled, v, logp)
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
                    prev_action = None  # 重置前一个动作
            
            if epoch < num_debug_epochs:
                print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} end")
            # 记录时间统计
            self.epoch_metrics['gpu_times'].append(epoch_gpu_time)
            self.epoch_metrics['cpu_times'].append(epoch_cpu_time)

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self._save_model(epoch)

            # Perform PPO update!
            if epoch < num_debug_epochs:
                print(f"Epoch {epoch} update start")
            self._update()
            if epoch < num_debug_epochs:
                print(f"Epoch {epoch} update end")

            # Log epoch info
            if epoch < num_debug_epochs:
                print(f"Epoch {epoch} log epoch info start")
            self._log_epoch_info(epoch, start_time)
            if epoch < num_debug_epochs:
                print(f"Epoch {epoch} log epoch info end")
        
        # Close TensorBoard writer
        self.tb_writer.close()

def ppo(env_fn, actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.1, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=100,
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
    
    # 网络参数控制
    parser.add_argument('--feature_dim', type=int, default=256, help='CNN特征维度')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 64], 
                       help='全连接层隐藏层大小')
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
        actor_critic = CNNActorCriticShared
        ac_kwargs = dict(
            feature_dim=args.feature_dim,
            actor_hidden=args.hidden_sizes,
            critic_hidden=args.hidden_sizes,
            car_racing_mode=True,
            use_framestack=True  # 启用FrameStack
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
