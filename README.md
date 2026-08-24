# Deep Reinforcement Learning - PPO From Scratch

A CUDA-based implementation of Proximal Policy Optimization (PPO) built from scratch using C++/cuda. This project focuses on understanding and implementing reinforcement learning algorithms at the systems level without relying on external deep learning frameworks.

This project is the next step after my CUDA neural network implementation, extending custom GPU-accelerated neural network components into a complete reinforcement learning pipeline.

## Video Demo

Add video demo here

## Overview

This project implements a reinforcement learning agent trained using the Proximal Policy Optimization (PPO) algorithm.

Instead of using existing deep learning libraries, the core components required for training are implemented manually, including neural network execution, policy optimization, value estimation, and GPU accelerated computation.

The goal is to explore how modern reinforcement learning algorithms work internally while building a complete RL system on top of a custom CUDA neural network backend.
## Enviroment

  A 2D car learns to drive to a specified target while avoiding obstacles, the Enviroment containing spawn point, target and obstacles randomizes every few generation for a generalised learning.

## Proximal Policy Optimization (PPO)

PPO is an actor-critic reinforcement learning algorithm designed to improve training stability by limiting how much the policy changes during each update.

This implementation uses:

- Actor-Critic architecture
- Policy network (Actor) for action selection
- Value network (Critic) for state value estimation
- PPO clipped surrogate objective
- Advantage estimation
- Return calculation
- Policy optimization
- Value function optimization
- Entropy regularization for exploration

## Training Pipeline

The PPO training process follows this workflow:

1. The current policy interacts with the environment and collects experience.
2. States, actions, rewards, and value predictions are stored.
3. Returns and advantages are calculated from collected experiences.
4. The Actor network is updated using the PPO clipped objective.
5. The Critic network is updated using value prediction loss.
6. The process repeats as the policy improves.

## CUDA Neural Network Foundation

This project builds on my previous CUDA neural network implementation:

CUDA Neural Network:
https://github.com/NobodyBuilds/cuda_neural_net

The neural network components created in that project are reused as the foundation for this reinforcement learning system.

By combining a custom CUDA neural network engine with PPO, this project creates a fully custom GPU accelerated reinforcement learning framework.

## Technologies

- C++
- CUDA
- Reinforcement Learning
- Proximal Policy Optimization (PPO)
- Actor-Critic Networks
- Neural Networks
- GPU Acceleration

## Why This Project

The purpose of this project is to understand reinforcement learning beyond using existing frameworks.

By implementing PPO from scratch, this project explores:

- How policies are optimized
- How value estimation guides learning
- How reinforcement learning algorithms interact with neural networks


## Project Progression

This project is part of a progression of building machine learning systems from the ground up:

1. CUDA Neural Network  
   Custom GPU accelerated neural network implementation.

2. Deep Reinforcement Learning  
   Applying the custom neural network backend to train agents using PPO.

3. Future Projects  
   Exploring more advanced reinforcement learning algorithms and GPU accelerated AI systems.

## Future Improvements

Possible improvements:

- More reinforcement learning algorithms
- Improved training visualization
- Additional environments
- Performance optimizations
- Better CUDA kernel utilization
- Adam optimizer

## Author

NobodyBuilds
