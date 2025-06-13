# PyTorch implementation of  D4PG 

This repository contains a PyTorch implementation of D4PG with IQN as the improved distributional Critic instead of C51. Also the extentions [Munchausen RL](https://arxiv.org/abs/2007.14430) and [D2RL](https://paperswithcode.com/paper/d2rl-deep-dense-architectures-in-1) are added and can be combined with D4PG as needed. 

**🆕 Update:** The repository has been updated to support **Gymnasium** (the successor to OpenAI Gym) while maintaining full backward compatibility.

#### Dependencies
Updated dependencies:
<pre>
Python 3.11+
PyTorch 1.4.0+  
Numpy 1.15.2+
gymnasium 0.28.0+
</pre>

Legacy dependencies (for reference):
<pre>
Python 3.6
PyTorch 1.4.0  
Numpy 1.15.2 
gym 0.10.11 
</pre>

## How to use:
The new script combines all extensions and the add-ons can be simply added by setting the corresponding flags.

`python run.py -info your_run_info`

**Parameter:**
To see the options:
`python run.py -h`


### Observe training results
  `tensorboard --logdir=runs`


Added Extensions:

- Prioritized Experience Replay [X]
- N-Step Bootstrapping [X]
- D2RL [X]
- Distributional IQN Critic [X]
- Munchausen RL [X]
- Parallel-Environments [X]

## Results 
### Environment: Pendulum

![Pendulum](imgs/D4PG_Improvements.png)

Below you can see how IQN reduced the variance of the Critic loss:

![CriticLoss](imgs/QvsIQN.png)


### Environment: LunarLander
![LunarLander](imgs/D4PG_LunarLanderContinuous.png)

Notes:

- Performance depends a lot on good hyperparameter->> tau for Per bigger (pendulum 1e-2) for regular replay (1e-3)

- BatchNorm had good impact on the overall performance (!)

## Gymnasium Migration

This repository has been successfully updated to support **Gymnasium** (the modern successor to OpenAI Gym). The migration:

- ✅ **Full compatibility** with gymnasium environments
- ✅ **Backward compatibility** maintained for existing workflows  
- ✅ **All extensions working**: PER, N-Step, Munchausen RL, D2RL, IQN
- ✅ **No performance impact** on training or evaluation

### Migration Details
- Updated environment API calls (`reset()`, `step()`, `seed()`)
- Enhanced vectorized environment support
- Improved error handling and type safety
- See `GYMNASIUM_MIGRATION.md` for detailed technical information

### Tested Environments
- ✅ `Pendulum-v1` 
- ✅ Other continuous control environments
- ✅ Vectorized multi-environment training



python run.py -env "Pendulum-v1" -frames 30000 -eval_every 1000 -eval_runs 1 -nstep 5 -learn_every 2 -per 1 -iqn 0 -w 4 -bs 512 -layer_size 128 -info "per_nstep_learnevery_t3_d2rl "  -t 1e-3 -d2rl 1