# D4PG-QR-FRM: Distributional Deep Deterministic Policy Gradient with Advanced Extensions

## Algorithm Overview

**D4PG (Distributional Deep Deterministic Policy Gradient)** is a state-of-the-art reinforcement learning algorithm that extends DDPG by incorporating distributional value learning. Instead of predicting scalar Q-values, D4PG learns the full distribution over returns, leading to more robust and sample-efficient learning.

### Key Features Implemented

This implementation extends the core D4PG algorithm with several advanced features:

- **🎯 Distributional Critic**: Uses **IQN (Implicit Quantile Networks)** instead of traditional C51 for improved distributional learning
- **🚀 Munchausen RL**: Entropy-regularized policy improvement for enhanced exploration and stability
- **🧠 Munchausen RL**: Enhanced exploration through better action-value estimation
- **⚡ Prioritized Experience Replay (PER)**: Importance sampling for more efficient learning from experience
- **🔄 N-Step Bootstrapping**: Multi-step returns for faster value propagation
- **🌊 Parallel Environments**: Vectorized environment support for accelerated training
- **🎮 Intrinsic Curiosity Module (ICM)**: Optional curiosity-driven exploration
- **💨 Performance Optimizations**: torch.compile, automatic mixed precision, and CPU/GPU optimizations

### D4PG Core Algorithm

The D4PG algorithm combines:

1. **Deterministic Policy Gradient**: Learn a deterministic policy μ(s) that maximizes expected return
2. **Distributional Value Learning**: Model the full return distribution Z(s,a) instead of just E[Z(s,a)]
3. **Target Networks**: Use slowly-updated target networks for stable learning
4. **Experience Replay**: Learn from past experiences stored in a replay buffer
5. **Ornstein-Uhlenbeck Noise**: Add temporally correlated noise for exploration

The distributional aspect allows the agent to model uncertainty in value estimates, leading to more robust decision-making and improved sample efficiency.

## Architecture Overview

The following diagram illustrates the complete architecture and data flow of the D4PG-QR-FRM implementation:

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#e1f5fe', 'primaryTextColor':'#000', 'primaryBorderColor':'#01579b', 'lineColor':'#666', 'fontSize':'16px', 'fontFamily':'Arial, sans-serif'}}}%%
graph TB
    %% Main execution entry points
    RunScript["`<b>run.py</b><br/>Main Training Script`"]
    EnjoyScript["`<b>enjoy.py</b><br/>Policy Evaluation`"]
    RunShell["`<b>run.sh</b><br/>Batch Training`"]
    
    %% Core Agent Components
    Agent["`<b>Agent</b><br/>(scripts/agent.py)<br/>• Coordinates training<br/>• Manages networks<br/>• Handles experience`"]
    
    %% Neural Networks
    subgraph Networks["<b>Neural Networks</b><br/>(scripts/networks.py)"]
        Actor["`<b>Actor Network</b><br/>Deterministic Policy<br/>μ(s) → a`"]
        
        Critic["`<b>Critic Network</b><br/>Standard Q-function<br/>Q(s,a) → scalar`"]
        
        IQN["`<b>IQN Network</b><br/>Implicit Quantile Network<br/>Distributional Q-learning`"]
        
        ActorTarget["`<b>Target Networks</b><br/>Slowly updated copies<br/>for stable learning`"]
    end
    
    %% Experience Replay Systems
    subgraph ReplayBuffer["<b>Experience Replay</b><br/>(scripts/replay_buffer.py)"]
        StandardReplay["`<b>ReplayBuffer</b><br/>Standard uniform sampling<br/>with n-step returns`"]
        
        PER["`<b>PrioritizedReplay</b><br/>Priority-based sampling<br/>using TD-errors`"]
    end
    
    %% Curiosity-Driven Exploration
    subgraph ICMModule["<b>Intrinsic Curiosity</b><br/>(scripts/ICM.py)"]
        ICM["`<b>ICM Module</b><br/>Curiosity-driven exploration`"]
        
        Forward["`<b>Forward Model</b><br/>Predicts next state<br/>from current state + action`"]
        
        Inverse["`<b>Inverse Model</b><br/>Predicts action<br/>from state transitions`"]
    end
    
    %% Environment & Training Loop
    subgraph Environment["<b>Environment Interaction</b>"]
        GymEnv["`<b>Gymnasium Environment</b><br/>(e.g., Pendulum-v1)<br/>Provides states, rewards`"]
        
        Noise["`<b>Exploration Noise</b><br/>• Ornstein-Uhlenbeck<br/>• Gaussian Noise`"]
    end
    
    %% Data Analysis & Monitoring
    subgraph Analysis["<b>Analysis & Monitoring</b>"]
        TensorBoard["`<b>TensorBoard</b><br/>Training metrics<br/>& visualizations`"]
        
        Notebook["`<b>Jupyter Notebook</b><br/>2FactorOUwJumps_Simulation.ipynb<br/>Statistical analysis`"]
        
        StatScript["`<b>statistical_analysis.py</b><br/>(Empty - for future analysis)`"]
    end
    
    %% Model Storage
    subgraph Storage["<b>Model Storage</b>"]
        RunsDir["`<b>runs/ directory</b><br/>• Model weights (.pth)<br/>• Hyperparameters (.json)<br/>• TensorBoard logs`"]
    end
    
    %% Data Flow Connections
    RunScript --> Agent
    EnjoyScript --> Agent
    RunShell --> RunScript
    
    Agent --> Actor
    Agent --> Critic
    Agent --> IQN
    Agent --> ActorTarget
    
    Agent --> StandardReplay
    Agent --> PER
    
    Agent --> ICM
    ICM --> Forward
    ICM --> Inverse
    
    Agent --> GymEnv
    Agent --> Noise
    
    Agent --> TensorBoard
    Agent --> RunsDir
    
    %% Algorithm Flow
    subgraph D4PGFlow["<b>D4PG Algorithm Flow</b>"]
        direction TB
        
        ObserveState["`<b>1. Observe State s</b>`"]
        SelectAction["`<b>2. Actor selects action a</b><br/>+ exploration noise`"]
        ExecuteAction["`<b>3. Execute action in environment</b><br/>observe s', r, done`"]
        StoreExperience["`<b>4. Store (s,a,r,s',done)</b><br/>in replay buffer`"]
        SampleBatch["`<b>5. Sample batch from buffer</b><br/>(prioritized or uniform)`"]
        ComputeTargets["`<b>6. Compute target values</b><br/>using target networks`"]
        UpdateCritic["`<b>7. Update Critic/IQN</b><br/>minimize distributional loss`"]
        UpdateActor["`<b>8. Update Actor</b><br/>maximize Q-value`"]
        UpdateTargets["`<b>9. Soft update target networks</b><br/>θ' ← τθ + (1-τ)θ'`"]
        UpdateICM["`<b>10. Update ICM (if enabled)</b><br/>for intrinsic rewards`"]
        
        ObserveState --> SelectAction
        SelectAction --> ExecuteAction
        ExecuteAction --> StoreExperience
        StoreExperience --> SampleBatch
        SampleBatch --> ComputeTargets
        ComputeTargets --> UpdateCritic
        UpdateCritic --> UpdateActor
        UpdateActor --> UpdateTargets
        UpdateTargets --> UpdateICM
        UpdateICM --> ObserveState
    end
    
    %% Key Features Annotations
    classDef mainComponent fill:#e1f5fe,stroke:#01579b,stroke-width:3px,font-size:16px
    classDef networkComponent fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,font-size:16px
    classDef dataComponent fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px,font-size:16px
    classDef analysisComponent fill:#fff3e0,stroke:#e65100,stroke-width:3px,font-size:16px
    
    class RunScript,Agent mainComponent
    class Actor,Critic,IQN,ActorTarget networkComponent
    class StandardReplay,PER,RunsDir dataComponent
    class TensorBoard,Notebook,StatScript analysisComponent
    
    %% Feature Annotations
    Agent -.->|"`<b>Key Features:</b><br/>• Distributional RL (IQN)<br/>• Munchausen RL<br/>• N-step returns<br/>• Prioritized Experience Replay<br/>• Intrinsic Curiosity (ICM)<br/>• torch.compile optimization<br/>• Mixed precision training`"| Networks
```

### Architecture Highlights

- **🏗️ Modular Design**: Each component can be enabled/disabled independently
- **🔄 Complete D4PG Flow**: Full implementation of the distributional policy gradient algorithm
- **🚀 Advanced Extensions**: State-of-the-art RL techniques integrated seamlessly
- **📊 Comprehensive Monitoring**: TensorBoard integration and analysis tools
- **⚡ Performance Optimized**: PyTorch 2.x features for maximum efficiency

The diagram shows how the main training script orchestrates the interaction between neural networks, experience replay, curiosity-driven exploration, and environment interaction, with comprehensive monitoring and storage capabilities.

## Dependencies

This implementation uses **Python 3.11** with the following core packages:

### Core Dependencies
```
Python 3.11.13
torch 2.8.0.dev20250613
numpy 2.2.6
gymnasium 1.0.0
```

### Scientific Computing & Visualization
```
matplotlib 3.10.3
pandas 2.3.0
plotly 6.1.2
seaborn 0.13.2
scienceplots 2.1.1
scipy 1.15.2
```

### Reinforcement Learning & Utilities
```
tensorboard 2.19.0
tqdm 4.67.1
psutil 7.0.0
cloudpickle 3.1.1
```

### Development & Profiling
```
ipython 9.3.0
jupyter_core 5.8.1
snakeviz 2.2.2
```

## Installation & Usage

### Quick Start
```bash
# Install dependencies (if using conda environment)
conda create -n d4pg python=3.11
conda activate d4pg

# Install required packages
pip install torch gymnasium matplotlib pandas plotly seaborn scipy tensorboard tqdm psutil cloudpickle

# Run training with default settings
python run.py -info "my_experiment"
```

### Configuration Options

All extensions can be easily enabled/disabled using command-line flags:

```bash
python run.py -h  # Show all available options
```

**Key Parameters:**
- `-env`: Environment name (default: "Pendulum-v1")
- `-frames`: Total training frames (default: 1,000,000)
- `-per 1`: Enable Prioritized Experience Replay
- `-munchausen 1`: Enable Munchausen RL
- `-iqn 1`: Use distributional IQN critic (default: 0)
- `-nstep N`: N-step bootstrapping (default: 1)
- `-w N`: Number of parallel environments (default: 1)
- `-learn_every N`: Learning frequency (default: 1)

### Baseline Configuration

Based on the current experiment setup in `run.sh`, the optimal baseline configuration is:

```bash
python run.py \
    -env="Pendulum-v1" \
    -frames=30000 \
    -eval_every=500 \
    -munchausen=1 \
    -eval_runs=5 \
    -nstep=5 \
    -learn_every=2 \
    -per=1 \
    -iqn=0 \
    -w=1 \
    -bs=64 \
    -layer_size=128 \
    -t=5e-4 \
    -info "Baseline4" \
    -seed 1
```

This configuration achieves:
- **Final Training Performance**: -123.790 episode return
- **Evaluation Performance**: -396.093 ± 190.691 (10 episodes)
- **Training Speed**: 537 FPS
- **Training Time**: ~56 seconds for 30,000 frames

### Monitoring Training

View real-time training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```


## Algorithm Extensions

This implementation includes several cutting-edge extensions that can be combined modularly:

### ✅ Implemented Extensions

- **Prioritized Experience Replay (PER)**: Samples important transitions more frequently based on TD-error magnitude
- **N-Step Bootstrapping**: Uses multi-step returns for faster value propagation and improved sample efficiency  
- **Distributional IQN Critic**: Implicit Quantile Networks for robust distributional value learning with reduced variance
- **Munchausen RL**: Entropy-regularized policy improvement that adds a scaled log-policy term to rewards for enhanced exploration
- **Parallel Environments**: Vectorized environment execution for faster data collection and training acceleration
- **Intrinsic Curiosity Module (ICM)**: Optional curiosity-driven exploration using prediction error as intrinsic reward
- **Performance Optimizations**: torch.compile acceleration, automatic mixed precision, and optimized CPU/GPU utilization

### Extension Benefits

- **PER + N-Step**: Dramatically improves sample efficiency by focusing on important experiences
- **Munchausen RL**: Provides natural exploration bonus and improved policy regularization
- **IQN Distributional Learning**: Reduced critic loss variance and more robust value estimates
- **Parallel Environments**: Linear speedup in data collection (4x workers = ~4x faster training)

## Performance Results

### Pendulum-v1 Environment

The current baseline configuration demonstrates strong performance on the Pendulum-v1 continuous control task:

**Training Metrics:**
- Environment: Pendulum-v1 (continuous control)
- Training Frames: 30,000
- Final Episode Return: -123.790
- Training Speed: 537 FPS
- Total Training Time: 55.97 seconds

**Evaluation Results (Final):**
- Average Return: -396.093 ± 190.691
- Best Episode: -121.875
- Worst Episode: -715.351
- Evaluation Episodes: 10

**Configuration Details:**
- Munchausen RL: Enabled
- Prioritized Experience Replay: Enabled  
- N-Step Bootstrapping: 5 steps
- Learning Frequency: Every 2 steps
- Batch Size: 64
- Network Size: 128 hidden units
- TAU (soft update): 5e-4

### Algorithm Improvements

![Pendulum](imgs/D4PG_Improvements.png)

The distributional IQN critic shows significant variance reduction compared to standard Q-learning:

![CriticLoss](imgs/QvsIQN.png)

### Additional Environments

![LunarLander](imgs/D4PG_LunarLanderContinuous.png)

**Performance Notes:**
- Hyperparameter sensitivity: TAU values should be larger for PER (1e-2) vs regular replay (1e-3)
- Batch normalization provides significant performance improvements
- Multi-environment training (workers > 1) scales training speed nearly linearly

## Gymnasium Migration & Compatibility

This repository has been successfully updated to support **Gymnasium** (the modern successor to OpenAI Gym) with full backward compatibility.

### Migration Features
- ✅ **Full Gymnasium compatibility** with all modern environment APIs
- ✅ **Backward compatibility** maintained for existing workflows  
- ✅ **All extensions verified**: PER, N-Step, Munchausen RL, IQN work seamlessly
- ✅ **Enhanced vectorized environments** with improved error handling
- ✅ **Zero performance impact** on training or evaluation speed

### Technical Improvements
- Updated environment API calls (`reset()`, `step()`, `seed()`)
- Enhanced vectorized environment support with proper gymnasium format handling
- Improved error handling and type safety throughout the codebase
- Automatic conversion between gymnasium (terminated, truncated) and legacy (done) formats

### Tested Environments
- ✅ **Pendulum-v1**: Primary testing environment with comprehensive benchmarks
- ✅ **Continuous control environments**: Full compatibility across gymnasium suite
- ✅ **Vectorized multi-environment training**: Parallel execution verified and optimized

### Example Usage
```bash
# Modern gymnasium environments
python run.py -env "Pendulum-v1" -frames 30000 -w 4 -info "gymnasium_test"

# Legacy naming still works
python run.py -env "LunarLanderContinuous-v2" -frames 50000 -info "legacy_compat"
```

## Project Structure

```
D4PG-QR-FRM/
├── run.py                    # Main training script with all configurations
├── enjoy.py                  # Model evaluation and visualization
├── run.sh                    # Baseline experiment configuration
├── scripts/
│   ├── agent.py             # D4PG agent with all extensions
│   ├── networks.py          # Actor/Critic networks (standard & deep)
│   ├── replay_buffer.py     # PER and standard replay buffer
│   ├── ICM.py               # Intrinsic Curiosity Module
│   └── MultiPro.py          # Vectorized environment wrapper
├── runs/                     # TensorBoard logs and saved models
└── __pycache__/             # Python bytecode cache
```

## References

- [D4PG: Distributed Distributional Deep Deterministic Policy Gradient](https://openreview.net/forum?id=SyZipzbCb)
- [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430)
- [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)