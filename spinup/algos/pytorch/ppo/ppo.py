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
# 视频录制相关导入
import cv2
from gymnasium.wrappers import RecordVideo
import shutil

DEFAULT_TBOARD_DIR = "/root/tf-logs" if osp.exists("/root/tf-logs") else osp.join(osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__)))),'../../data')
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__)))),'../../data')
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
    def __init__(self, feature_dim, act_dim, hidden_sizes=(256,128), init_log_std=0.3):
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
        log_std = self.log_std.clamp(-2, 2)
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

    def step(self, obs, return_env_action=True, deterministic=False):
        """
        obs: (B,3,H,W) or (3,H,W). Returns:
          - action np.array (tanh space for training consistency)
          - value np.array
          - logp np.array (for the exact action fed back into policy gradient)
        If return_env_action=True and car_racing_mode, returns mapped env action.
        If deterministic=True, uses mean action instead of sampling.
        """
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            pi, feats = self._pi_dist(obs)
            if self.is_box:
                if deterministic:
                    # 确定性动作：使用均值
                    mu, log_std = self.pi(feats)
                    a_tanh = torch.tanh(mu)  # 确定性tanh均值
                    v = self.v(feats)
                    logp = pi.log_prob(a_tanh)  # 计算确定性动作的log_prob
                else:
                    # 随机动作：采样
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
                if deterministic:
                    # 离散动作的确定性选择：选择概率最大的动作
                    logits = self.policy_logits(feats)
                    a = torch.argmax(logits, dim=-1)
                else:
                    a = pi.sample()
                v = self.v(feats)
                logp = pi.log_prob(a)
                return a.cpu().numpy(), v.cpu().numpy(), logp.cpu().numpy()

    def act(self, obs, return_env_action=True, deterministic=False):
        return self.step(obs, return_env_action=return_env_action, deterministic=deterministic)[0]

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


class OffRoadEarlyTerminate(gym.Wrapper):
    def __init__(
        self,
        env,
        green_threshold=(60, 120, 60),   # 绿色像素阈值 (R<G_th[1], G>G_th[1], B<G_th[1]) 简易版本
        region_rel=(0.55, 0.75, 0.35, 0.65),  # 检测区域在图像中的相对坐标 (y1,y2,x1,x2)
        offroad_ratio_thresh=0.60,      # 区域内绿色像素比例超过此阈值判定离路
        end_on_offroad=True,            # 是否直接结束 episode
        offroad_penalty=-5.0,           # 触发离路时附加一次性惩罚
        min_steps_before_check=50       # 起步若干帧内不做检测，避免出场阶段误判
    ):
        super().__init__(env)
        self.green_threshold = green_threshold
        self.region_rel = region_rel
        self.offroad_ratio_thresh = offroad_ratio_thresh
        self.end_on_offroad = end_on_offroad
        self.offroad_penalty = offroad_penalty
        self.min_steps_before_check = min_steps_before_check
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _get_last_frame(self, obs):
        # obs 可能是 (H,W,C) 或 (S,H,W,C)（FrameStack）
        if obs.ndim == 4:  # (S,H,W,C)
            frame = obs[-1]
        else:
            frame = obs
        return frame

    def _is_offroad(self, obs):
        frame = self._get_last_frame(obs)  # uint8 HWC
        H, W, C = frame.shape
        y1r, y2r, x1r, x2r = self.region_rel
        y1, y2 = int(H * y1r), int(H * y2r)
        x1, x2 = int(W * x1r), int(W * x2r)
        roi = frame[y1:y2, x1:x2]  # (h, w, 3)

        # 更宽松的绿色检测：降低阈值，更容易检测到绿色
        R = roi[..., 0].astype(np.int32)
        G = roi[..., 1].astype(np.int32)
        B = roi[..., 2].astype(np.int32)

        # 更宽松的绿色判断：G明显高于R和B
        green_mask = (G > R + 20) & (G > B + 20) & (G > 80)
        
        # 也可以检测"非道路"区域：检测草地、泥土等
        # 检测高亮度的非灰色区域（可能是草地）
        bright_non_gray = (G + R + B > 200) & (abs(G - R) > 30) & (abs(G - B) > 30)
        
        # 综合判断：绿色区域或高亮非灰色区域
        offroad_mask = green_mask | bright_non_gray
        offroad_ratio = offroad_mask.mean()
        
        return offroad_ratio > self.offroad_ratio_thresh, float(offroad_ratio)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1

        # 默认不在前若干步检查，避免刚出发时视觉不稳定
        if self._t >= self.min_steps_before_check:
            offroad, offroad_ratio = self._is_offroad(obs)
            info['offroad_green_ratio'] = offroad_ratio
            if offroad:
                # 附加惩罚（一次性）
                reward = reward + self.offroad_penalty
                if self.end_on_offroad:
                    terminated = True
                    info['early_terminated_offroad'] = True

        return obs, reward, terminated, truncated, info

def make_env():
    env = gym.make('CarRacing-v3')             # 先创建原环境
    # env = OffRoadEarlyTerminate(env,           # 再加离路提前结束
    #                             offroad_penalty=-5.0,
    #                             end_on_offroad=True,
    #                             min_steps_before_check=50,        # 减少起步检测延迟
    #                             # 优化检测参数 - 更宽松的设置
    #                             region_rel=(0.6, 0.9, 0.2, 0.8),  # 检测更大区域
    #                             offroad_ratio_thresh=0.7,          # 提高阈值，减少误判
    #                             green_threshold=(50, 100, 50))     # 调整绿色检测阈值
    return env


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.9):
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
        
        # 统计GAE数值（规范化前）- 每10个epoch打印一次
        if proc_id() == 0:
            current_epoch = getattr(self, '_current_epoch', 0)
            if current_epoch % 10 == 0:
                self._print_gae_statistics(all_adv, all_ret, all_deltas)
        
        # 直接归一化优势函数，不进行winsorize和增强
        adv_mean, adv_std = mpi_statistics_scalar(all_adv)
        all_adv = (all_adv - adv_mean) / adv_std
        
        # 验证规范化后的优势函数 - 每10个epoch打印一次
        if proc_id() == 0:
            current_epoch = getattr(self, '_current_epoch', 0)
            if current_epoch % 10 == 0:
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
        
        if proc_id() != 0:
            return
            
        print(f"\n📊 GAE统计信息:")
        print(f"  优势函数 (裁剪前):")
        print(f"    均值: {adv.mean():.6f}")
        print(f"    标准差: {adv.std():.6f}")
        print(f"    最小值: {adv.min():.6f}")
        print(f"    最大值: {adv.max():.6f}")
        
        # 检查是否有极端值
        extreme_positive = np.sum(adv > 5.0)
        extreme_negative = np.sum(adv < -5.0)
        if extreme_positive > 0 or extreme_negative > 0:
            print(f"    极端值: {extreme_positive} 个 > 5.0, {extreme_negative} 个 < -5.0")
            print(f"    💡 注意：已移除winsorize裁剪，直接进行标准化")
        
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
                 steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.15, pi_lr=3e-4,
                 vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
                 target_kl=0.05, save_freq=100, device=None, min_steps_per_proc=None, record_videos=False,
                 ent_coef=0.005):

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
        self.record_videos = record_videos
        self.ent_coef = ent_coef
        
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
        self._setup_agent()
        self._setup_training_components()
    
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
        
        # Set up TensorBoard writer (使用单独的TensorBoard目录)
        self.tb_output_dir = osp.join(DEFAULT_TBOARD_DIR, exp_name)
        self.tb_writer = SummaryWriter(log_dir=self.tb_output_dir)
        
        # Save configuration to JSON file
        config_path = os.path.join(self.output_dir, 'config.json')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tb_output_dir, exist_ok=True)
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
        
        # 打印关键参数信息
        if proc_id() == 0:
            print(f"🔧 训练参数:")
            print(f"   max_ep_len: {self.max_ep_len}")
            print(f"   steps_per_epoch: {self.steps_per_epoch}")
            print(f"   steps_per_epoch: {self.steps_per_epoch}")

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
            print("Epoch    | Return    | Policy Loss | Value Loss | KL        | Entropy  | ClipFrac  | Avg Length | Early Stop | GPU Time | CPU Time | GPU Memory")
            print("=" * 180)

    def _setup_training_components(self):
        """Setup training components"""
        num_procs_val = num_procs()

        self.local_steps_per_epoch = self.steps_per_epoch
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.local_steps_per_epoch, self.gamma, self.lam)

        # Set up optimizers for policy and value function
        # 策略优化器优化encoder+pi，确保encoder参与策略学习
        self.pi_optimizer = Adam(list(self.ac.encoder.parameters()) + list(self.ac.pi.parameters()), lr=self.pi_lr)
        # 价值优化器也优化encoder+v，让critic能适配视觉表征
        self.vf_optimizer = Adam(list(self.ac.encoder.parameters()) + list(self.ac.v.parameters()), lr=self.vf_lr)
        
        # 设置学习率调度器 - warmup + cosine
        self._setup_lr_schedulers()

        self.minibatch_size = 2048  # 如果总样本少于 2048，可设为 512 或 1024
        self.policy_epochs = 5
        self.value_epochs = 4

        self.kl_history = []
        self.cf_history = []
    
    def _setup_lr_schedulers(self):
        """设置warmup+cosine学习率调度器"""
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        # Warmup步数：前10%的epoch进行warmup
        warmup_epochs = max(1, int(0.1 * self.epochs))
        self.warmup_epochs = warmup_epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Warmup阶段：线性增长
                return epoch / warmup_epochs
            else:
                # Cosine阶段：从warmup结束到训练结束
                progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        # 为策略和价值函数分别设置调度器
        self.pi_scheduler = LambdaLR(self.pi_optimizer, lr_lambda)
        self.vf_scheduler = LambdaLR(self.vf_optimizer, lr_lambda)
        
        if proc_id() == 0:
            print(f"📈 学习率调度: warmup={warmup_epochs} epochs, cosine={self.epochs-warmup_epochs} epochs")

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
        
        # 添加熵正则化
        ent_coef = getattr(self, 'ent_coef', 0.005)  # 默认熵系数
        entropy = pi.entropy().mean()
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - ent_coef * entropy

        # 使用更稳健的KL近似（仅作监控）
        robust_kl = 0.5 * ((logp - logp_old) ** 2).mean().item()
        ent = entropy.item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        
        # 详细调试信息 - 每10个epoch打印一次
        current_epoch = getattr(self.buf, '_current_epoch', 0)

        pi_info = dict(kl=robust_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def _compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        obs = obs.to(self.device); ret = ret.to(self.device)
        feats = self.ac.encoder(obs)  # 移除detach，让critic优化也更新encoder
        v = self.ac.v(feats)
        return F.smooth_l1_loss(v, ret)

    def _save_model(self, epoch):
        """Save model at specified epoch"""
        model_path = os.path.join(self.output_dir, f'model_epoch_{epoch}.pth')
        torch.save(self.ac.state_dict(), model_path)
        
        # 只在进程0且启用视频录制时进行推理评测和视频录制
        if proc_id() == 0 and self.record_videos:
            self._evaluate_and_record_videos(epoch, model_path)

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
                mpi_avg_grads(self.ac.encoder)  # 添加encoder梯度平均
                mpi_avg_grads(self.ac.v)
                self.vf_optimizer.step()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # 记录本 epoch 的 KL/CF
        mean_kl = float(np.mean(kl_list_epoch)) if kl_list_epoch else 0.0
        mean_cf = float(np.mean(cf_list_epoch)) if cf_list_epoch else 0.0
        self.kl_history.append(mean_kl)
        self.cf_history.append(mean_cf)
        if len(self.kl_history) > 20:
            self.kl_history = self.kl_history[-20:]
            self.cf_history = self.cf_history[-20:]

        # 写入指标（注意：这里用 old 的 pi_l_old/v_l_old 作为 epoch 级损失参考）
        ent_log = pi_info_old['ent'] if isinstance(pi_info_old, dict) and 'ent' in pi_info_old else 0.0
        self.epoch_metrics['loss_pi'].append(pi_l_old)
        self.epoch_metrics['loss_v'].append(v_l_old)
        self.epoch_metrics['kl'].append(mean_kl)
        self.epoch_metrics['entropy'].append(ent_log)
        self.epoch_metrics['clip_frac'].append(mean_cf)
        self.epoch_metrics['stop_iter'].append(0)  # 不再用迭代计数作为早停标志
        
        # 更新学习率调度器（在优化器步骤之后）
        self.pi_scheduler.step()
        self.vf_scheduler.step()

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
        clip_frac = np.mean(self.epoch_metrics['clip_frac']) if self.epoch_metrics['clip_frac'] else 0.0
        avg_length = np.mean(self.epoch_metrics['ep_lengths']) if self.epoch_metrics['ep_lengths'] else 0.0
        early_stop = np.mean(self.epoch_metrics['stop_iter']) if self.epoch_metrics['stop_iter'] else 0.0
        early_stop_flag = "True" if early_stop < self.train_pi_iters - 1 else "False"
        
        gpu_time = self.epoch_metrics['gpu_times'][-1] if self.epoch_metrics['gpu_times'] else 0.0
        cpu_time = self.epoch_metrics['cpu_times'][-1] if self.epoch_metrics['cpu_times'] else 0.0
        total_time = gpu_time + cpu_time
        gpu_ratio = (gpu_time / total_time * 100) if total_time > 0 else 0        
        time_info = f" | GPU: {gpu_time:.2f}s({gpu_ratio:.1f}%)"
        
        # 单行打印，严格对齐（Return使用原始奖励，与评估一致）
        print(f"Epoch {epoch:4d} | Return: {ep_return:5.2f} | Policy Loss: {policy_loss:5.4f} | Value Loss: {value_loss:5.4f} | KL: {kl_div:8.4f} | Entropy: {entropy:5.4f} | ClipFrac: {clip_frac:5.4f} | Avg Length: {avg_length:6.1f} | Early Stop: {early_stop_flag:5s}{time_info}")
        
        # 记录到 TensorBoard - 基本训练指标
        self.tb_writer.add_scalar('Training/Epoch', epoch, epoch)
        self.tb_writer.add_scalar('Training/Environment_Interactions', (epoch + 1) * self.steps_per_epoch, epoch)
        self.tb_writer.add_scalar('Training/Time', time.time() - start_time, epoch)
        
        # 记录奖励和回合信息
        if self.epoch_metrics['ep_returns']:
            # 记录原始奖励（与评估一致）
            self.tb_writer.add_scalar('Reward/Episode_Return_Raw', np.mean(self.epoch_metrics['ep_returns']), epoch)
            if len(self.epoch_metrics['ep_returns']) > 1:
                self.tb_writer.add_scalar('Reward/Episode_Return_Raw_Std', np.std(self.epoch_metrics['ep_returns']), epoch)
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
        
        # 记录学习率
        current_pi_lr = self.pi_optimizer.param_groups[0]['lr']
        current_vf_lr = self.vf_optimizer.param_groups[0]['lr']
        self.tb_writer.add_scalar('Learning_Rate/Policy_LR', current_pi_lr, epoch)
        self.tb_writer.add_scalar('Learning_Rate/Value_LR', current_vf_lr, epoch)
        
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
        ep_ret, ep_len = 0, 0  # 原始奖励和长度
        ep_scaled_ret = 0  # 缩放后奖励
        prev_action = None  # 用于转向平滑

        # Main loop: collect experience in env and update/log each epoch
        num_debug_epochs = 0
        num_debug_steps = 3
        reward_scale = getattr(self, 'reward_scale', 3.0)
        for epoch in range(self.epochs):
            # 设置当前epoch信息，供PPOBuffer使用
            self.buf._current_epoch = epoch
            
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
                ep_ret += r  # 原始奖励
                r_scaled = r / reward_scale  # 缩放后奖励
                ep_scaled_ret += r_scaled  # 累计缩放后奖励
                if epoch < num_debug_epochs and t < num_debug_steps:
                    print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} next_o {next_o.shape} r {r} terminated {terminated} truncated {truncated}")
                cpu_end = time.time()
                epoch_cpu_time += (cpu_end - cpu_start)
                
                # 更新前一个动作用于转向平滑
                prev_action = action_for_env.copy()
                
                d = terminated or truncated  # 环境终止: 自然终止 OR 截断终止
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
                if proc_id() == 0 and epoch < num_debug_epochs:
                    print(f"🔍 Episode {len(self.epoch_metrics['ep_returns'])+1} ep_len {ep_len} max_ep_len {self.max_ep_len} timeout {timeout} terminal {terminal} epoch_ended {epoch_ended}")
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
                        # print("自然终止")
                    self.buf.finish_path(v)  # (obs, act, rew, val, logp) -> (obs, act, ret, adv, logp, adv, ret)
                    if epoch < num_debug_epochs and t < num_debug_steps:
                        print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} finish_path")
                    if terminal:
                        # 只有自然终止时才记录，其他情况不记录
                        self.epoch_metrics['ep_returns'].append(ep_ret)  # 使用原始奖励
                        self.epoch_metrics['ep_lengths'].append(ep_len)
                    o, _ = self.env.reset()
                    if epoch < num_debug_epochs and t < num_debug_steps:
                        print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} reset")
                    ep_ret, ep_len = 0, 0  # 重置原始奖励和长度
                    ep_scaled_ret = 0  # 重置缩放后奖励
                    prev_action = None  # 重置前一个动作
            
            if epoch < num_debug_epochs:
                print(f"Epoch {epoch} step {t}/{self.local_steps_per_epoch} end")
            # 记录时间统计
            self.epoch_metrics['gpu_times'].append(epoch_gpu_time)
            self.epoch_metrics['cpu_times'].append(epoch_cpu_time)

            # Save model
            if (epoch % self.save_freq == 10) or (epoch == self.epochs-1):
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
    
    def _evaluate_and_record_videos(self, epoch, model_path):
        """评估模型并录制视频"""
        import os
        import shutil
        import glob
        
        print(f"🎬 开始评估模型并录制视频 (Epoch {epoch})")
        
        # 创建视频保存目录
        video_dir = os.path.join(os.path.dirname(model_path), f'videos_epoch_{epoch}')
        os.makedirs(video_dir, exist_ok=True)
        
        # 创建评估环境，使用与训练相同的环境配置
        import gymnasium as gym
        # 重新创建环境，直接指定render_mode
        base_env = gym.make('CarRacing-v3', render_mode='rgb_array')
        # eval_env = OffRoadEarlyTerminate(base_env,           # 再加离路提前结束
        #                                 offroad_penalty=-5.0,
        #                                 end_on_offroad=True,
        #                                 min_steps_before_check=50,        # 减少起步检测延迟
        #                                 region_rel=(0.6, 0.9, 0.2, 0.8),  # 检测更大区域
        #                                 offroad_ratio_thresh=0.7,          # 提高阈值，减少误判
        #                                 green_threshold=(50, 100, 50))     # 调整绿色检测阈值
        
        # 为CarRacing环境添加FrameStack
        eval_env = FrameStack(base_env, stack_size=4)
        
        # 录制5段视频
        num_episodes = 2
        episode_returns = []
        
        for episode in range(num_episodes):
            print(f"  录制第 {episode + 1}/{num_episodes} 段视频...")
            
            # 为每个episode创建独立的视频目录，避免RecordVideo冲突
            episode_video_dir = os.path.join(video_dir, f'episode_{episode + 1}_temp')
            os.makedirs(episode_video_dir, exist_ok=True)
            
            # 直接在RecordVideo环境上运行，获取真实回报
            try:
                # 创建视频录制环境（使用独立目录）
                env_with_video = RecordVideo(
                    eval_env, 
                    video_folder=episode_video_dir,
                    episode_trigger=lambda x: True,  # 每个episode都录制
                    name_prefix='video',  # 简单名称
                    video_length=1000  # 最大录制长度
                )
                
                # 运行episode并录制
                obs, _ = env_with_video.reset()
                episode_return = 0
                episode_length = 0
                done = False
                
                while not done and episode_length < 1000:
                    with torch.no_grad():
                        # 处理观测
                        if len(obs.shape) == 3:  # 单帧图像
                            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                        elif len(obs.shape) == 4:  # FrameStack
                            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                        else:
                            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                        
                        # 获取确定性动作 - 使用与训练一致的动作映射逻辑
                        a_tanh, _, _ = self.ac.step(obs_tensor, return_env_action=False, deterministic=True)
                        a_tanh_tensor = torch.as_tensor(a_tanh, dtype=torch.float32)
                        action = self.ac._map_to_carracing(a_tanh_tensor).cpu().numpy()
                        
                        # 确保动作形状正确
                        if len(action.shape) > 1 and action.shape[0] == 1:
                            action = action[0]
                    
                    # 执行动作
                    obs, reward, terminated, truncated, _ = env_with_video.step(action)
                    done = terminated or truncated
                    
                    episode_return += reward
                    episode_length += 1
                
                # 重要：关闭环境以触发视频保存
                env_with_video.close()
                
                # 等待视频文件写入完成
                import time
                time.sleep(0.5)  # 等待500ms让视频文件写入完成
                
                episode_returns.append(episode_return)
                print(f"    Episode {episode + 1}: Return = {episode_return:.2f}, Length = {episode_length}")
                
                # 移动并重命名视频文件为最终名称
                try:
                    # 调试：列出临时目录中的所有文件
                    if os.path.exists(episode_video_dir):
                        all_files = os.listdir(episode_video_dir)
                        print(f"    调试: 临时目录 {episode_video_dir} 中的文件: {all_files}")
                    
                    # 尝试多种文件模式
                    patterns = [
                        os.path.join(episode_video_dir, 'video-episode-*.mp4'),
                        os.path.join(episode_video_dir, '*.mp4'),
                        os.path.join(episode_video_dir, 'video-*.mp4')
                    ]
                    
                    video_files = []
                    for pattern in patterns:
                        video_files = glob.glob(pattern)
                        if video_files:
                            print(f"    调试: 找到文件，模式: {pattern}")
                            break
                    
                    if video_files:
                        old_path = video_files[0]
                        final_filename = f'episode_{episode + 1}_return={episode_return:.2f}_length={episode_length}.mp4'
                        new_path = os.path.join(video_dir, final_filename)
                        
                        # 移动文件到主目录并重命名
                        shutil.move(old_path, new_path)
                        print(f"    视频文件: {final_filename}")
                        
                        # 清理临时目录
                        try:
                            os.rmdir(episode_video_dir)
                        except:
                            pass  # 忽略清理失败
                    else:
                        print(f"    警告: 未找到episode {episode + 1}的视频文件")
                        print(f"    调试: 临时目录: {episode_video_dir}")
                        print(f"    调试: 目录存在: {os.path.exists(episode_video_dir)}")
                        if os.path.exists(episode_video_dir):
                            print(f"    调试: 目录内容: {os.listdir(episode_video_dir)}")
                        
                except Exception as rename_error:
                    print(f"    移动视频文件失败: {rename_error}")
                
            except Exception as e:
                print(f"    Episode {episode + 1} 录制失败: {e}")
                episode_returns.append(0.0)
            
            finally:
                # env_with_video已经在episode结束后关闭了
                pass
        
        # 统计结果
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        max_return = np.max(episode_returns)
        min_return = np.min(episode_returns)
        
        print(f"📊 评估结果 (Epoch {epoch}):")
        print(f"  平均奖励: {mean_return:.2f} ± {std_return:.2f}")
        print(f"  最高奖励: {max_return:.2f}")
        print(f"  最低奖励: {min_return:.2f}")
        print(f"  视频保存目录: {video_dir}")
        print(f"  🎯 使用确定性动作评估（减少随机性）")
        
        # 记录到TensorBoard
        self.tb_writer.add_scalar('Evaluation/Mean_Return', mean_return, epoch)
        self.tb_writer.add_scalar('Evaluation/Std_Return', std_return, epoch)
        self.tb_writer.add_scalar('Evaluation/Max_Return', max_return, epoch)
        self.tb_writer.add_scalar('Evaluation/Min_Return', min_return, epoch)
        
        eval_env.close()

def ppo(env_fn, actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.15, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.1, save_freq=100, device=None, min_steps_per_proc=None, record_videos=False,
        ent_coef=0.005):
    agent = PPOAgent(env_fn, actor_critic, ac_kwargs, seed, steps_per_epoch, epochs, 
                    gamma, clip_ratio, pi_lr, vf_lr, train_pi_iters, train_v_iters, 
                    lam, max_ep_len, target_kl, save_freq, device, min_steps_per_proc, record_videos,
                    ent_coef)
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
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
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
    parser.add_argument('--record_videos', action='store_true', 
                       help='是否在保存checkpoint时录制视频')
    parser.add_argument('--save_freq', type=int, default=100, help='保存模型的频率')
    parser.add_argument('--max_ep_len', type=int, default=1000, help='每个episode的最大步数')
    parser.add_argument('--ent_coef', type=float, default=0.005, help='熵正则化系数')
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

    ppo(lambda : make_env(), actor_critic=actor_critic,
        ac_kwargs=ac_kwargs, gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters, target_kl=args.target_kl,
        max_ep_len=args.max_ep_len, device=device,
        min_steps_per_proc=args.min_steps_per_proc,
        save_freq=args.save_freq,
        record_videos=args.record_videos,
        ent_coef=args.ent_coef)
