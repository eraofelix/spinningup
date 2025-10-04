#!/usr/bin/env python3
"""
Simple training curve plotting script (English version to avoid font issues)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_training_curves(log_dir):
    """Plot training curves"""
    progress_file = os.path.join(log_dir, 'progress.txt')
    
    if not os.path.exists(progress_file):
        print(f"Progress file not found: {progress_file}")
        return
    
    # Read training data
    data = pd.read_csv(progress_file, sep='\t')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # 1. Reward curves
    axes[0, 0].plot(data['Epoch'], data['AverageEpRet'], label='Training Reward', color='blue')
    if 'AverageTestEpRet' in data.columns:
        axes[0, 0].plot(data['Epoch'], data['AverageTestEpRet'], label='Test Reward', color='red')
    axes[0, 0].set_title('Reward Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Q-value curves
    if 'QVals' in data.columns:
        axes[0, 1].plot(data['Epoch'], data['QVals'], color='green')
        axes[0, 1].set_title('Q-Value Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Q-Value')
        axes[0, 1].grid(True)
    
    # 3. Policy loss
    if 'LossPi' in data.columns:
        axes[1, 0].plot(data['Epoch'], data['LossPi'], color='orange')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # 4. Q-network loss
    if 'LossQ' in data.columns:
        axes[1, 1].plot(data['Epoch'], data['LossQ'], color='purple')
        axes[1, 1].set_title('Q-Network Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print training statistics
    print("\n=== Training Statistics ===")
    print(f"Total epochs: {data['Epoch'].iloc[-1]}")
    print(f"Final training reward: {data['AverageEpRet'].iloc[-1]:.2f}")
    if 'AverageTestEpRet' in data.columns:
        print(f"Final test reward: {data['AverageTestEpRet'].iloc[-1]:.2f}")
    print(f"Best training reward: {data['AverageEpRet'].max():.2f}")
    if 'AverageTestEpRet' in data.columns:
        print(f"Best test reward: {data['AverageTestEpRet'].max():.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # Default to latest DDPG results
        log_dir = "/Users/kun/code/spinningup/data/ddpg/ddpg_s0"
    
    plot_training_curves(log_dir)
