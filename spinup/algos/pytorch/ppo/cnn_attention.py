import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Box, Discrete


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


def count_vars(module):
    """计算模块参数数量"""
    return sum([np.prod(p.shape) for p in module.parameters()])


# 测试函数
def test_cnn_actor_critic():
    """测试CNN Actor-Critic网络"""
    import gymnasium as gym
    
    # 创建CarRacing环境
    env = gym.make('CarRacing-v3', render_mode='rgb_array')
    obs_space = env.observation_space
    act_space = env.action_space
    
    print(f"观测空间: {obs_space}")
    print(f"动作空间: {act_space}")
    
    # 创建网络
    ac = CNNActorCritic(obs_space, act_space)
    
    # 测试前向传播
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    print(f"输入观测形状: {obs_tensor.shape}")
    
    # 测试step方法
    action, value, logp = ac.step(obs_tensor)
    print(f"动作: {action}")
    print(f"价值: {value}")
    print(f"对数概率: {logp}")
    
    # 计算参数数量
    pi_params = count_vars(ac.pi)
    v_params = count_vars(ac.v)
    total_params = count_vars(ac)
    
    print(f"策略网络参数: {pi_params}")
    print(f"价值网络参数: {v_params}")
    print(f"总参数: {total_params}")
    
    env.close()


if __name__ == '__main__':
    test_cnn_actor_critic()
