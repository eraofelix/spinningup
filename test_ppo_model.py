#!/usr/bin/env python3
"""
测试训练好的 PPO 模型
"""
import torch
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import os
import argparse
from spinup.algos.pytorch.ppo.core import MLPActorCritic

def detect_model_architecture(model_path):
    """自动检测模型架构"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 分析策略网络的权重
    pi_weights = {k: v for k, v in checkpoint.items() if k.startswith('pi.')}
    
    # 找到所有线性层的权重
    linear_layers = []
    for key in pi_weights.keys():
        if 'weight' in key and 'mu_net' in key:
            layer_num = int(key.split('.')[2])  # 获取层号
            weight_shape = pi_weights[key].shape
            linear_layers.append((layer_num, weight_shape))
    
    # 按层号排序
    linear_layers.sort(key=lambda x: x[0])
    
    # 调试信息
    print(f"检测到的线性层: {linear_layers}")
    
    # 根据权重形状推断隐藏层大小
    hidden_sizes = []
    for i, (layer_num, shape) in enumerate(linear_layers):
        if i == 0:  # 第一层：输入层 -> 隐藏层
            hidden_sizes.append(shape[0])
        elif i == len(linear_layers) - 1:  # 最后一层：输出层
            continue
        else:  # 中间层：隐藏层
            hidden_sizes.append(shape[0])
    
    # 如果检测失败，尝试从错误信息推断
    if len(hidden_sizes) == 0 or len(hidden_sizes) != len(linear_layers) - 1:
        print("检测失败，尝试从权重形状推断...")
        # 从权重形状推断隐藏层大小
        hidden_sizes = []
        for i, (layer_num, shape) in enumerate(linear_layers):
            if i == 0:  # 第一层
                hidden_sizes.append(shape[0])
            elif i == len(linear_layers) - 1:  # 最后一层
                continue
            else:  # 中间层
                hidden_sizes.append(shape[0])
    
    print(f"检测到模型架构: {hidden_sizes}")
    
    return hidden_sizes

def load_model(model_path, env, hidden_sizes=None):
    """加载训练好的模型"""
    # 如果没有指定架构，自动检测
    if hidden_sizes is None:
        hidden_sizes = detect_model_architecture(model_path)
    
    # 创建网络结构
    ac = MLPActorCritic(env.observation_space, env.action_space, 
                       hidden_sizes=hidden_sizes)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    ac.load_state_dict(checkpoint)
    
    return ac

def test_model(env_name, model_path, num_episodes=5, render=True, hidden_sizes=None):
    """测试模型性能"""
    # 创建环境
    env = gym.make(env_name, render_mode='human' if render else None)
    
    # 加载模型（自动检测架构）
    ac = load_model(model_path, env, hidden_sizes=hidden_sizes)
    ac.eval()  # 设置为评估模式
    
    print(f"开始测试模型: {model_path}")
    print(f"环境: {env_name}")
    print(f"测试回合数: {num_episodes}")
    print("=" * 50)
    
    episode_returns = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False
        
        while not done:
            # 获取动作（确定性策略，不添加噪声）
            with torch.no_grad():
                action = ac.act(torch.as_tensor(obs, dtype=torch.float32))
            
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}: Return = {episode_return:.2f}, Length = {episode_length}")
    
    env.close()
    
    # 统计结果
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    max_return = np.max(episode_returns)
    min_return = np.min(episode_returns)
    
    print("=" * 50)
    print("测试结果统计:")
    print(f"平均奖励: {mean_return:.2f} ± {std_return:.2f}")
    print(f"最高奖励: {max_return:.2f}")
    print(f"最低奖励: {min_return:.2f}")
    print(f"测试回合数: {num_episodes}")

def find_latest_model(experiment_dir):
    """找到最新的模型文件"""
    model_files = []
    for file in os.listdir(experiment_dir):
        if file.startswith('model_epoch_') and file.endswith('.pth'):
            model_files.append(file)
    
    if not model_files:
        raise FileNotFoundError(f"在 {experiment_dir} 中没有找到模型文件")
    
    # 按epoch编号排序，返回最新的
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(experiment_dir, model_files[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 PPO 模型")
    parser.add_argument('--env', type=str, default='HalfCheetah-v5', help='环境名称')
    parser.add_argument('--model_path', type=str, help='模型文件路径')
    parser.add_argument('--experiment_dir', type=str, help='实验目录（自动找最新模型）')
    parser.add_argument('--episodes', type=int, default=5, help='测试回合数')
    parser.add_argument('--no_render', action='store_true', help='不显示渲染')
    parser.add_argument('--hid', type=int, help='隐藏层神经元数量（可选，自动检测）')
    parser.add_argument('--l', type=int, help='隐藏层数量（可选，自动检测）')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    elif args.experiment_dir:
        model_path = find_latest_model(args.experiment_dir)
        print(f"找到最新模型: {model_path}")
    else:
        # 默认使用最新的实验目录
        import glob
        experiment_dirs = glob.glob('data/ppo_*')
        if not experiment_dirs:
            print("错误: 没有找到实验目录")
            exit(1)
        
        latest_dir = max(experiment_dirs, key=os.path.getctime)
        model_path = find_latest_model(latest_dir)
        print(f"使用最新实验目录: {latest_dir}")
        print(f"找到最新模型: {model_path}")
    
    # 创建隐藏层大小（如果指定了参数）
    hidden_sizes = None
    if args.hid is not None and args.l is not None:
        hidden_sizes = [args.hid] * args.l
        print(f"使用指定的模型架构: {hidden_sizes}")
    else:
        print("自动检测模型架构...")
    
    # 测试模型
    test_model(
        env_name=args.env,
        model_path=model_path,
        num_episodes=args.episodes,
        render=not args.no_render,
        hidden_sizes=hidden_sizes
    )

