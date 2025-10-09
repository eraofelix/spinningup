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
import sys
sys.path.append('spinup')
from spinup.algos.pytorch.ppo.ppo import CNNActorCriticShared
from gymnasium.wrappers import FrameStackObservation as FrameStack

def detect_model_architecture(model_path):
    """自动检测模型架构（支持MLP和CNN）"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 检查是否是CNN模型 - 检查encoder层
    has_cnn_layers = any('encoder' in key for key in checkpoint.keys())
    
    if has_cnn_layers:
        print("🖼️  检测到CNN模型")
        return detect_cnn_architecture(checkpoint)
    else:
        print("📊 检测到MLP模型")
        return detect_mlp_architecture(checkpoint)

def detect_cnn_architecture(checkpoint):
    """检测CNN模型架构"""
    print("🔍 分析模型权重...")
    
    # 检测特征维度 - 从encoder.head层
    feature_dim = 256  # 默认值
    for key, weight in checkpoint.items():
        if 'encoder.head' in key and 'weight' in key:
            weight_shape = weight.shape
            if len(weight_shape) == 2:  # 线性层权重
                feature_dim = weight_shape[1]  # 输入维度
                print(f"  检测到特征维度: {feature_dim}")
                break
    
    # 检测策略网络隐藏层 - 从pi.mlp层
    pi_weights = {k: v for k, v in checkpoint.items() if 'pi.mlp' in k and 'weight' in key}
    hidden_sizes = []
    for key in sorted(pi_weights.keys()):
        weight_shape = pi_weights[key].shape
        if len(weight_shape) == 2:  # 线性层权重
            hidden_sizes.append(weight_shape[0])
    
    # 移除最后一层（输出层）
    if len(hidden_sizes) > 1:
        hidden_sizes = hidden_sizes[:-1]
    
    print(f"  策略网络隐藏层: {hidden_sizes}")
    
    # 检测价值网络隐藏层
    v_weights = {k: v for k, v in checkpoint.items() if 'v.v' in k and 'weight' in key}
    critic_hidden = []
    for key in sorted(v_weights.keys()):
        weight_shape = v_weights[key].shape
        if len(weight_shape) == 2:  # 线性层权重
            critic_hidden.append(weight_shape[0])
    
    # 移除最后一层（输出层）
    if len(critic_hidden) > 1:
        critic_hidden = critic_hidden[:-1]
    
    print(f"  价值网络隐藏层: {critic_hidden}")
    
    # 根据实际检测到的架构调整参数
    if not hidden_sizes:
        hidden_sizes = [256, 128]  # 默认值
    if not critic_hidden:
        critic_hidden = [256, 128]  # 默认值
    
    return {
        'type': 'cnn',
        'feature_dim': feature_dim,
        'actor_hidden': hidden_sizes if hidden_sizes else [256, 128],
        'critic_hidden': critic_hidden if critic_hidden else [256, 128]
    }

def detect_mlp_architecture(checkpoint):
    """检测MLP模型架构"""
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
    
    # 根据权重形状推断隐藏层大小
    hidden_sizes = []
    for i, (layer_num, shape) in enumerate(linear_layers):
        if i == 0:  # 第一层：输入层 -> 隐藏层
            hidden_sizes.append(shape[0])
        elif i == len(linear_layers) - 1:  # 最后一层：输出层
            continue
        else:  # 中间层：隐藏层
            hidden_sizes.append(shape[0])
    
    print(f"MLP隐藏层大小: {hidden_sizes}")
    
    return {
        'type': 'mlp',
        'hidden_sizes': hidden_sizes
    }

def load_model(model_path, env, architecture_info=None):
    """加载训练好的模型（支持MLP和CNN）"""
    # 对于CarRacing环境，直接使用默认的CNN架构
    if 'CarRacing' in str(env.spec.id) if hasattr(env, 'spec') and env.spec else 'CarRacing' in str(type(env)):
        print("🖼️  创建CNN网络（CarRacing默认架构）...")
        ac = CNNActorCriticShared(
            env.observation_space, 
            env.action_space,
            feature_dim=256,  # 使用默认值
            actor_hidden=[256, 128],  # 使用默认值
            critic_hidden=[256, 128],  # 使用默认值
            car_racing_mode=True,
            use_framestack=True
        )
        
        # 尝试加载模型权重
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            ac.load_state_dict(checkpoint)
            print("✅ 模型权重加载成功")
        except RuntimeError as e:
            print(f"❌ 模型权重加载失败: {e}")
            print("💡 尝试使用兼容的架构...")
            
            # 如果加载失败，尝试使用检测到的架构
            if architecture_info is None:
                architecture_info = detect_model_architecture(model_path)
            
            ac = CNNActorCriticShared(
                env.observation_space, 
                env.action_space,
                feature_dim=architecture_info.get('feature_dim', 256),
                actor_hidden=architecture_info.get('actor_hidden', [256, 128]),
                critic_hidden=architecture_info.get('critic_hidden', [256, 128]),
                car_racing_mode=True,
                use_framestack=True
            )
            
            checkpoint = torch.load(model_path, map_location='cpu')
            ac.load_state_dict(checkpoint)
    else:
        print("📊 创建MLP网络...")
        # 暂时不支持MLP，因为当前代码只支持CNN
        raise NotImplementedError("当前测试脚本只支持CNN模型")
    
    return ac

def test_model(env_name, model_path, num_episodes=5, render=True, architecture_info=None):
    """测试模型性能（支持MLP和CNN）"""
    # 创建环境
    env = gym.make(env_name, render_mode='human' if render else None)
    
    # 为CarRacing环境添加FrameStack（如果使用CNN模型）
    if 'CarRacing' in env_name and architecture_info is None:
        # 自动检测模型类型
        temp_checkpoint = torch.load(model_path, map_location='cpu')
        has_cnn_layers = any('encoder' in key for key in temp_checkpoint.keys())
        if has_cnn_layers:
            print("🏎️  检测到CarRacing环境，添加FrameStack(4)包装器")
            env = FrameStack(env, stack_size=4)
            print(f"📊 FrameStack后观测空间: {env.observation_space}")
    
    # 加载模型（自动检测架构）
    ac = load_model(model_path, env, architecture_info=architecture_info)
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
                # 处理图像观测
                if len(obs.shape) == 3:  # 图像观测 (H, W, C)
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = ac.act(obs_tensor)
                    # 确保动作是正确的形状：从 (1, 3) 转换为 (3,)
                    if len(action.shape) > 1 and action.shape[0] == 1:
                        action = action[0]  # 取第一个（也是唯一的）动作
                elif len(obs.shape) == 4:  # FrameStack观测 (S, H, W, C)
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = ac.act(obs_tensor)
                    # 确保动作是正确的形状：从 (1, 3) 转换为 (3,)
                    if len(action.shape) > 1 and action.shape[0] == 1:
                        action = action[0]  # 取第一个（也是唯一的）动作
                else:  # 向量观测
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
    # 网络参数（可选，自动检测）
    parser.add_argument('--feature_dim', type=int, help='CNN特征维度（可选，自动检测）')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', help='全连接层隐藏层大小（可选，自动检测）')
    
    # MLP网络参数（向后兼容）
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
    
    # 创建架构信息（如果指定了参数）
    architecture_info = None
    if any([args.feature_dim, args.hidden_sizes, args.hid, args.l]):
        print("使用指定的模型架构参数...")
        architecture_info = {}
        
        # 判断是CNN还是MLP参数
        if args.feature_dim or 'CarRacing' in args.env:
            # CNN参数
            architecture_info['type'] = 'cnn'
            if args.feature_dim:
                architecture_info['feature_dim'] = args.feature_dim
            if args.hidden_sizes:
                architecture_info['actor_hidden'] = args.hidden_sizes
                architecture_info['critic_hidden'] = args.hidden_sizes
        elif args.hid and args.l:
            # MLP参数（向后兼容）
            architecture_info['type'] = 'mlp'
            architecture_info['hidden_sizes'] = [args.hid] * args.l
        elif args.hidden_sizes:
            # 只有hidden_sizes，需要判断环境类型
            if 'CarRacing' in args.env:
                architecture_info['type'] = 'cnn'
                architecture_info['actor_hidden'] = args.hidden_sizes
                architecture_info['critic_hidden'] = args.hidden_sizes
            else:
                architecture_info['type'] = 'mlp'
                architecture_info['hidden_sizes'] = args.hidden_sizes
        
        print(f"指定架构: {architecture_info}")
    else:
        print("自动检测模型架构...")
    
    # 测试模型
    test_model(
        env_name=args.env,
        model_path=model_path,
        num_episodes=args.episodes,
        render=not args.no_render,
        architecture_info=architecture_info
    )

