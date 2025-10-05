import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import gymnasium_robotics
import time
# MPI imports will be done conditionally when needed
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import scipy.signal
from gymnasium.spaces import Box, Discrete
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# Constants moved from user_config.py
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__)))),'data')
FORCE_DATESTAMP = False


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false, 

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in 
    ``spinup/user_config.py``. 

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


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


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        # Move tensors to CPU before converting to numpy
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


# ============================================================================
# PPO Buffer and Agent classes
# ============================================================================

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self, use_mpi=True):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        if use_mpi:
            from spinup.utils.mpi_tools import mpi_statistics_scalar
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        else:
            adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class PPOAgent:
    """
    Proximal Policy Optimization (by clipping) Agent
    
    with early stopping based on approximate KL
    """
    
    def __init__(self, env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
                 steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
                 target_kl=0.05, logger_kwargs=dict(), save_freq=100, use_mpi=True, device=None):
        """
        Initialize PPO Agent
        
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with a 
                ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
                module.

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to PPO.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy objective.

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to take 
                on policy loss per epoch.

            train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
                
            use_mpi (bool): Whether to use MPI for parallel training. If False,
                runs in single-process mode.
        """
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
        self.logger_kwargs = logger_kwargs
        self.save_freq = save_freq
        self.use_mpi = use_mpi
        
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
        if self.use_mpi:
            # Import MPI modules only when needed
            from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi
            from spinup.utils.mpi_tools import proc_id
            
            # Special function to avoid certain slowdowns from PyTorch + MPI combo.
            print(f"🔧 进程 {proc_id()}: 开始设置 PyTorch MPI...")
            try:
                setup_pytorch_for_mpi()
                print(f"✅ 进程 {proc_id()}: PyTorch MPI 设置完成")
            except Exception as e:
                print(f"❌ 进程 {proc_id()}: PyTorch MPI 设置失败: {e}")
                raise
        else:
            print("🚫 禁用MPI模式: 跳过MPI设置")

        # 创建带时间戳的输出目录
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.logger_kwargs['output_dir']
        # 如果目录以 ppo_s0 结尾，替换为带时间戳的目录
        if base_dir.endswith('ppo_s0'):
            base_dir = base_dir.replace('ppo_s0', f'ppo_{timestamp}')
        else:
            # 如果目录不以 ppo_s0 结尾，在末尾添加时间戳
            base_dir = f"{base_dir}_{timestamp}"
        
        # 更新 logger_kwargs 中的 output_dir
        self.logger_kwargs['output_dir'] = base_dir
        
        # Set up TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.logger_kwargs['output_dir'])
        
        # Save configuration to JSON file
        config_path = os.path.join(self.logger_kwargs['output_dir'], 'config.json')
        os.makedirs(self.logger_kwargs['output_dir'], exist_ok=True)
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
        seed = self.seed # + 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        self.env = self.env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

    def _setup_agent(self):
        """Setup actor-critic agent"""
        # Create actor-critic module
        self.ac = self.actor_critic(self.env.observation_space, self.env.action_space, **self.ac_kwargs)
        
        # Move model to device (GPU/CPU)
        self.ac = self.ac.to(self.device)
        print(f"📱 模型已移动到设备: {self.device}")

        # Sync params across processes (only if using MPI)
        if self.use_mpi:
            from spinup.utils.mpi_pytorch import sync_params
            sync_params(self.ac)

        # Count variables (only for first process or single process mode)
        if not self.use_mpi:
            # Single process mode
            var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.v])
            print(f'\nNumber of parameters: pi: {var_counts[0]}, v: {var_counts[1]}')
            print("=" * 180)
            print("Epoch    | Return    | Policy Loss | Value Loss | KL        | Entropy  | Early Stop | GPU Time | CPU Time | GPU Memory")
            print("=" * 180)
        else:
            # MPI mode
            from spinup.utils.mpi_tools import proc_id
            if proc_id() == 0:
                var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.v])
                print(f'\nNumber of parameters: pi: {var_counts[0]}, v: {var_counts[1]}')
                print("=" * 180)
                print("Epoch    | Return    | Policy Loss | Value Loss | KL        | Entropy  | Early Stop | GPU Time | CPU Time | GPU Memory")
                print("=" * 180)

    def _setup_training_components(self):
        """Setup training components"""
        # Set up experience buffer
        if self.use_mpi:
            from spinup.utils.mpi_tools import num_procs
            self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
        else:
            self.local_steps_per_epoch = self.steps_per_epoch
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
        """Compute value function loss"""
        obs, ret = data['obs'], data['ret']
        
        # Move data to device
        obs = obs.to(self.device)
        ret = ret.to(self.device)
        
        return ((self.ac.v(obs) - ret)**2).mean()

    def _save_model(self, epoch):
        """Save model at specified epoch"""
        model_path = os.path.join(self.logger_kwargs['output_dir'], f'model_epoch_{epoch}.pth')
        torch.save(self.ac.state_dict(), model_path)

    def _update(self):
        """Perform PPO update"""
        # 测量GPU训练时间
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_train_start = time.time()
        
        data = self.buf.get(self.use_mpi)

        pi_l_old, pi_info_old = self._compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(data)
            if self.use_mpi:
                from spinup.utils.mpi_tools import mpi_avg
                kl = mpi_avg(pi_info['kl'])
            else:
                kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            if self.use_mpi:
                from spinup.utils.mpi_pytorch import mpi_avg_grads
                mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(data)
            loss_v.backward()
            if self.use_mpi:
                from spinup.utils.mpi_pytorch import mpi_avg_grads
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
        # Print epoch info (only for first process or single process mode)
        if not self.use_mpi:
            # Single process mode
            self._print_epoch_info(epoch, start_time)
        else:
            # MPI mode
            from spinup.utils.mpi_tools import proc_id
            if proc_id() == 0:
                self._print_epoch_info(epoch, start_time)
    
    def _print_epoch_info(self, epoch, start_time):
        """Print epoch information"""
        # 计算平均值
        ep_return = np.mean(self.epoch_metrics['ep_returns']) if self.epoch_metrics['ep_returns'] else 0.0
        policy_loss = np.mean(self.epoch_metrics['loss_pi']) if self.epoch_metrics['loss_pi'] else 0.0
        value_loss = np.mean(self.epoch_metrics['loss_v']) if self.epoch_metrics['loss_v'] else 0.0
        kl_div = np.mean(self.epoch_metrics['kl']) if self.epoch_metrics['kl'] else 0.0
        entropy = np.mean(self.epoch_metrics['entropy']) if self.epoch_metrics['entropy'] else 0.0
        
        # 检查是否有早停
        early_stop = np.mean(self.epoch_metrics['stop_iter']) if self.epoch_metrics['stop_iter'] else 0.0
        early_stop_flag = "True" if early_stop < self.train_pi_iters - 1 else "False"
        
        # GPU性能监控和时间统计
        gpu_info = ""
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**2
            gpu_info = f" | GPU: {gpu_memory:.1f}MB/{gpu_max_memory:.1f}MB"
        
        # 时间统计
        gpu_time = self.epoch_metrics['gpu_times'][-1] if self.epoch_metrics['gpu_times'] else 0.0
        cpu_time = self.epoch_metrics['cpu_times'][-1] if self.epoch_metrics['cpu_times'] else 0.0
        total_time = gpu_time + cpu_time
        gpu_ratio = (gpu_time / total_time * 100) if total_time > 0 else 0
        cpu_ratio = (cpu_time / total_time * 100) if total_time > 0 else 0
        
        time_info = f" | GPU: {gpu_time:.2f}s({gpu_ratio:.1f}%) | CPU: {cpu_time:.2f}s({cpu_ratio:.1f}%)"
        
        # 单行打印，严格对齐
        print(f"Epoch {epoch:4d} | Return: {ep_return:8.2f} | Policy Loss: {policy_loss:8.4f} | Value Loss: {value_loss:8.4f} | KL: {kl_div:8.4f} | Entropy: {entropy:8.4f} | Early Stop: {early_stop_flag:5s}{time_info}{gpu_info}")
        
        # 记录到 TensorBoard
        # 基本训练指标
        self.tb_writer.add_scalar('Training/Epoch', epoch, epoch)
        self.tb_writer.add_scalar('Training/Environment_Interactions', (epoch + 1) * self.steps_per_epoch, epoch)
        self.tb_writer.add_scalar('Training/Time', time.time() - start_time, epoch)
        
        # 记录奖励
        if self.epoch_metrics['ep_returns']:
            self.tb_writer.add_scalar('Reward/Episode_Return', np.mean(self.epoch_metrics['ep_returns']), epoch)
            if len(self.epoch_metrics['ep_returns']) > 1:
                self.tb_writer.add_scalar('Reward/Episode_Return_Std', np.std(self.epoch_metrics['ep_returns']), epoch)
        
        # 记录回合长度
        if self.epoch_metrics['ep_lengths']:
            self.tb_writer.add_scalar('Episode/Length', np.mean(self.epoch_metrics['ep_lengths']), epoch)
        
        # 记录价值估计
        if self.epoch_metrics['v_vals']:
            self.tb_writer.add_scalar('Values/Value_Estimates', np.mean(self.epoch_metrics['v_vals']), epoch)
        
        # 记录损失
        if self.epoch_metrics['loss_pi']:
            self.tb_writer.add_scalar('Loss/Policy_Loss', np.mean(self.epoch_metrics['loss_pi']), epoch)
        if self.epoch_metrics['loss_v']:
            self.tb_writer.add_scalar('Loss/Value_Loss', np.mean(self.epoch_metrics['loss_v']), epoch)
        
        # 记录策略指标
        if self.epoch_metrics['kl']:
            self.tb_writer.add_scalar('Policy/KL_Divergence', np.mean(self.epoch_metrics['kl']), epoch)
        if self.epoch_metrics['entropy']:
            self.tb_writer.add_scalar('Policy/Entropy', np.mean(self.epoch_metrics['entropy']), epoch)
        if self.epoch_metrics['clip_frac']:
            self.tb_writer.add_scalar('Policy/ClipFraction', np.mean(self.epoch_metrics['clip_frac']), epoch)
        
        # 记录训练指标
        if self.epoch_metrics['stop_iter']:
            self.tb_writer.add_scalar('Training/StopIterations', np.mean(self.epoch_metrics['stop_iter']), epoch)
        
        # 记录时间统计
        if self.epoch_metrics['gpu_times']:
            self.tb_writer.add_scalar('Performance/GPU_Time', self.epoch_metrics['gpu_times'][-1], epoch)
        if self.epoch_metrics['cpu_times']:
            self.tb_writer.add_scalar('Performance/CPU_Time', self.epoch_metrics['cpu_times'][-1], epoch)
        
        # 记录GPU内存使用
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            self.tb_writer.add_scalar('Performance/GPU_Memory_MB', gpu_memory, epoch)
        
        # 清空当前 epoch 的数据，为下一个 epoch 做准备
        for key in self.epoch_metrics:
            self.epoch_metrics[key] = []
        
        # Log model parameters to TensorBoard
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
        for epoch in range(self.epochs):
            # 批量收集数据以提高GPU效率
            batch_obs = []
            batch_actions = []
            batch_rewards = []
            batch_values = []
            batch_logps = []
            
            # 时间统计变量
            epoch_gpu_time = 0.0
            epoch_cpu_time = 0.0
            
            for t in range(self.local_steps_per_epoch):
                # GPU计算时间测量
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()  # 确保GPU操作完成
                gpu_start = time.time()
                
                # Move observation to device
                obs_tensor = torch.as_tensor(o, dtype=torch.float32).to(self.device)
                a, v, logp = self.ac.step(obs_tensor)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()  # 确保GPU操作完成
                gpu_end = time.time()
                epoch_gpu_time += (gpu_end - gpu_start)

                # CPU环境交互时间测量
                cpu_start = time.time()
                next_o, r, terminated, truncated, _ = self.env.step(a)
                cpu_end = time.time()
                epoch_cpu_time += (cpu_end - cpu_start)
                
                d = terminated or truncated
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, a, r, v, logp)
                # 记录价值估计到 TensorBoard 指标中
                self.epoch_metrics['v_vals'].append(v)
                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t==self.local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal) and (not self.use_mpi or proc_id() == 0):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        # GPU计算时间测量
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        gpu_start = time.time()
                        obs_tensor = torch.as_tensor(o, dtype=torch.float32).to(self.device)
                        _, v, _ = self.ac.step(obs_tensor)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        gpu_end = time.time()
                        epoch_gpu_time += (gpu_end - gpu_start)
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # 记录到 TensorBoard 指标中
                        self.epoch_metrics['ep_returns'].append(ep_ret)
                        self.epoch_metrics['ep_lengths'].append(ep_len)
                    o, _ = self.env.reset()
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


def ppo(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.05, logger_kwargs=dict(), save_freq=100, use_mpi=True, device=None):
    """
    Proximal Policy Optimization (by clipping) function for backward compatibility
    
    This function creates a PPOAgent and calls its train method.
    """
    agent = PPOAgent(env_fn, actor_critic, ac_kwargs, seed, steps_per_epoch, epochs, 
                    gamma, clip_ratio, pi_lr, vf_lr, train_pi_iters, train_v_iters, 
                    lam, max_ep_len, target_kl, logger_kwargs, save_freq, use_mpi, device)
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
    parser.add_argument('--no-mpi', action='store_true', help='禁用MPI，使用单进程模式')
    parser.add_argument('--device', type=str, default=None, help='指定设备 (cuda/cpu/auto)')
    args = parser.parse_args()

    # 根据参数决定是否使用MPI
    use_mpi = not args.no_mpi
    
    # 处理设备参数
    device = args.device
    if device == 'auto':
        device = None  # 让代码自动检测
    elif device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，回退到CPU")
        device = 'cpu'
    
    if use_mpi:
        from spinup.utils.mpi_tools import mpi_fork
        mpi_fork(args.cpu)  # run parallel code with mpi
    else:
        print("🚫 禁用MPI模式: 使用单进程训练")

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters, target_kl=args.target_kl,
        logger_kwargs=logger_kwargs, use_mpi=use_mpi, device=device)
