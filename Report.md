[//]: # (Image References)

[image1]: ./img/averaged_scores_train.jpg "Averaged training scores"
[image2]: ./img/averaged_scores_test.jpg "Averaged test scores"

# Continuus Control Project - Report

## Overview

- The goal of this project was to train an agent that controls a double-jointed arm in order to target locations.
- I used slightly modified version of DDPG algorithm. The main modification is that the collecting episodes and training the agents are being done asynchronically. Additinally I used uniformly random noise instead of Ornstein-Uhlenbeck noise.
- As a base for my code I used the [Udacity DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) implementation.
- It took 432 episodes to reach the goal (100 episode window average score >= 30), but I used 2,000 episodes for simulations.
- This code uses a Unity environment provided by Udacity. For more information check the `README.md` where the installation process is described.


## Learning Algorithm

#### DDPG (original algorithm)

DDPG is a modification of DQN algorithm destined for solving continuous control tasks. The algorithm was introduced by T. Lillicrap et. al. in 2015 and it is described [here](https://arxiv.org/abs/1509.02971).

In DDQN we have two networks.
- The Actor network that is responsible for maping the policy function $State\rightarrow Action$
- The Critic network that is responsible for estimating State-action values $Q(S, a)$

Both networks have two copies, (local and target). The local networks are updated using the
loss functions, whereas the target networks are updated using the soft update rule:
$target weights = (1-\tau)*target weights + \tau * local weights$


The original DDQN algorithm looks as follows:

- Initialize Critic and Actor Networks
- Initialize local copies of Critic and Actor Networks
- Initialize replay buffer
- Repeat for N episodes:
    - receive the initial state
    - repeat for t steps (or until the episode is finished):
        - select an action acording to the current policy (with noise)
        - execute ation, observe reward and next state
        - store (state, action, reward, next_state) in replay buffer
        - sample a minibatch from the replay buffer
        - update (local) critic and actor networks
        - update target networks using soft updates.


**Noise**

In order to ensure exploration, the [Ornstein-Uhlenbeck process](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823) noise is added to each action during training. The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction, which results in temporally correlated values centered around 0.


#### Parallelized DDPG (implemented solution)
Although DDPG seems to be very clear, there is a drawback in it. Namely, it is not parallelized. Between each learning step agent has to wait for one episode steps, and - while the agent is learning - the episodes are not generated. As the agent uses replay buffer and DDPG is an off-policy algorithm, there is no need to wait for a response from the environment before the next training steps. I simply parallelized these to steps, so my algorithm looks as follows:

*Initialization*
- Initialize Critic and Actor Networks
- Initialize local copies of Critic and Actor Networks
- Initialize replay buffer

*Thread 1 - Interacting with environment*
- Repeat for N episodes
    - receive the initial state
    - repeat for t steps (or until the episode is finished):
        - select an action acording to the current policy (with noise)
        - execute ation, observe reward and next state
        - store (state, action, reward, next_state) in replay buffer

*Thread 2 - Learning*
- Repeat as long as episodes are generated
    - sample a minibatch from the replay buffer
    - update (local) critic and actor networks
    - update target networks using soft updates.

This modification eliminates the delays that occur due to the sequential implementation of the algorithm.

*Note: I haven't done the literature review, so I do not know if my modification has a name or an author. This idea just came to me during the implementation. Parallelization here is very straightforward, so there probably is a paper that introduced it.*


**Noise**

Although the temporally correlated noise looks cool, I do not think it is necessary to use it in this particular example. The learning batches are randomly selected from the Replay Buffer, so the advantages of temporal correlation are very limited, and such a correlation can introduce additional bias. I changed the noise to uncorrelated uniformly random distribution and it speeds up the learning process. The noise was computed according to the formula:

<code>
    sigma * np.random.rand(len(action)) - sigma / 2
</code>

where the coefficient $\sigma$ (sigma) is the standard deviation of the noise.

The noise serves one more purpose for this particular task. It helps with keeping the robot arm near to the center of the ball. The 0.04 reward is granted if the arm touches the ball, regardless of their exact position. Without the noise, the agent could be happy enough to keep the arm on the verge of the ball, which is unstable. The noise awards the agent if it tries to keep the arm in the center of the ball.

In the experiments, I discovered that the small noise speeds up learning, whereas the large noise helps the agent to work more stable (keep the robot's arm in the center of the ball). Thus, while the agent was working better, the noise was increased according to the formula:

$\sigma=\frac{max(0, desired score-averaged score)}{desired score}*(\sigma_{max}-\sigma_{min})$



#### Neural Networks
**Critic**
A variation of the `Deep and Wide` network was used as a critic network. The parameters are as follows:
- Input layer (1): State vector, 33 dimensions
- First layer: 256 neurons
- Input layer (2): Action vector, 4 dimentions
- Second layer: 128 neurons. Inputs to second layer are outputs from the first layer concatenated with the action vector
- Third layer: 64 neurons
- Output layer: 1 neuron

The output layer uses linear activation function, all other activation functions are ReLU. The action vector is not connected to the first layer. In other words, the first layer is responsible for preprocessing the state vector only. I took this approach from the original Udacity implementation. I do not think it has some crucial impact on the results or performance, but it also should not harm anything, so I just kept it as is. No additional experiments were contucted to justify it.

**Actor**

A fully connected (dense) neural network were used, as a critic and actor networks.
with the following parameters:
- Input layer (1): State vector, 33 dimensions
- First layer: 256 neurons, ReLU activation function.
- Second layer: 128 neurons, ReLU activation function.
- Third layer: 64 neurons, ReLU activation function.
- Output layer - 4 neurons, Tanh activation function. (4 neurons correspond to 4 dimensions of the action vector).

The $tanh$ activation function at the output layer is used for clipping. In this environment, all the actions shall be scaled from -1 to 1, and this is what $tanh$ is doing.

#### Hyperparameters
As a base I took  the hyperparameters from the DQN exercise, and modified them slightly. Namely:
- I increased the replay buffer size from 100k to 500k. This is the result of experiments. If the buffer size is too small model doesn't learn very well, as there is not enough examples (it remains overfitting). If the buffer is too large, model learn much slower as it cannot fully leverage the most up-to-date SARS tupples.
- I increased the batch size to 512 to speed up the learning and get more from my GPU.

Parameters from `dqn_agent.py`
<code>
BUFFER_SIZE = int(5e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
</code>

Parameters from `generate_episode`
<code>
n_episodes=2000       # Number of episodes
print_every=100       # how often the information shall be printed (n. of episodes)
save_every=100        # how often the models shall be saved (n. of episodes)
noise_max = 0.1       # maximum sigma for noise generation
noise_min = 0.05      # minimum sigma for noise generation
min_noise_level = 35  # averaged score that corresponds to maximum sigma for noise generation
</code>

## Implementation

#### Code decsription
The [Udacity DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) implementation was used as a base code, with only minor changes. The structure of the files is as follows:

- `models.py` - contains the actor and critic pytorch models
- `ddpg_agent.py` - contains the gist of the algorithm namely:
    - `OUNoise` class is responsible for generating noise that is added to actions to ensure exporation. As this noise is temporarily correlated, it is natural to implement it in a separate class.
    - `ReplayBuffer` class that can store and sample experience tupples
    - `Agent` class - the main class with the following methods:
        - `step` - the method from the original code that samples 1 batch from `ReplayBuffer` and performs one learning step. Due to parallelization I didn't use this method
        - `act` - the method that takes states as input and returns the corresponding action, according to the current policy. 
        - `reset` - resets noise parameters (useful with OU noise, not used in my solution)
        - `learn` - performs actor and critic learning for one batch
        - `soft_update` - updates target networks (called by `learn`)
- `Continuous_Control.ipynb` - notebook with training code. Points 1-3 of this notebook are loading and presenting the environment. The task starts in point 4:
    - `train_agent` trains an agent. To start training the Replay Buffer must be largest than the batch size. Training is performed as long as the global variable `run_training == True`
    - `generate_episodes` generates episodes. To keep everything as similar to Udacity example as possible, this method is also responsible for collecting printing the on-going training results and saving the models.  
- `TestAgents.ipynb` - notebook that tests already trained agents.
- `AnalyzeResults.ipynb` - notebook that analyzes the training and testing results.
- `models/checkpoint_actor_500.pth` - final actor network 
- `models/checkpoint_critic_500.pth` - final critic network 

#### Experiment description
Only the final (working) experiment is described here.
2000 episodes were generated, which corresponds to 2,002,000 steps (each episode ends after 1,001 steps). In the meantime 271,342 learning steps has been performed which gives 7.38 episode steps / learning step.

The training scores averaged with moving average (window size = 100 episodes) are presented in the figure below. 
![Scores][image1]
All the metrics in the chart are computed for the last 100 episodes.
- Red line stands for the mean score.
- Blue area stands for the standard deviation of the scores.
- Green area stands for the range of 90th percentile (top 5% and bottom 5% of scores are deleted).

The `avg_score==30` theshold was achieved after 432 episodes. 

For the agent saved after 500 episodes, the training scores are as follows:
- Mean score: 37.15
- Std dev.: 2.04
- Scores above 30: 99%


#### Tests
Although the goal was to achieve the averaged reward, the results may be misleading due to the noise. For example, if we decrease the noise with time to very small values it is easy to achieve the average reward equal to 35 relatively quickly. Thus, in order to check the real behavior of the agent, I decided to perform no-noise tests.
During the training the models were saved every 100 iterations. In order to test the agents, the models were loaded and used to generate the  average score for 100 episodes. To generate the average score the optimal policy (without noise) were used.
The results are presented in the figure below.
![Scores][image2]

It turns out that starting from the model based on 300 episodes all the models met the project requirements. No other surprises here.

For the agent saved after 500 episodes, the testing scores are as follows:
- Mean score: 36.24
- Std dev.: 2.58
- Scores above 30: 98%



## Final model selection
Every model trained 500 episodes or longer meets the requirement. As a final model I choose the first one that met the requirements, that is the model based on 500 episodes.


## Conclusions
1. The parallelization turned out to be very helpful. The expected threshold was achieved after 432 episodes. As 1 learning step corresponded to 7.39 environment steps, we can compare the speed with the 'classic' DDPG algorithm. 432 parallel episodes correspond to 59 episodes in the 'classic' approach. I started my experiments with 'classic' DDPG and I do not think it is doable to meet the goal in 59 steps.
2. The size of the replay buffer was a surprisingly important parameter. With too small replay buffer the model doesn't learn very well (it looks like some kind of overfitting). With too large replay buffer the model learns too slowly.
3. I spent 3 days experimenting with the noise. I managed to meet the goal of this exercise using the noise that follows Ornstein-Uhlenbeck process, uniformly random noise, normal distributed noise and without the noise (the randomness of the environment is enough to ensure exploration). I am still not sure if I fully understand the all the nuances of the noise here, but I am sure that the noise affected both speed of learning and the robustness of the presented solution.   
4. In the code, the entire training has been performed in one loop. I did it to simplify the code, but the incremental solution makes more sense. Namely - if you see that the learning process is not good enough you should be capable of manipulating learning parameters in the middle of the process.
5. Obtaining the expected average result (30) was relatively easy, but if you want to push your results above it, it becomes more difficult. I think that there is still a room for improvement in the reward definition. For example, each robot movement can be additionally penalized with a small negative reward, in order to minimize the unnecessary movements of the robot arm. Also, the positive reward should depend on the proximity of the robot arm to the center of the ball. I did not explore it, but this may probably be achieved in two ways. One way is to modify the environment. The other way is to compute the final reward using the rewards provided by the environment and the state vector.

## Ideas for future work
- Learn how to create/modify Unity environments. 
- Make a literature review to check if the parallel solution was already introduced. If not, try to repeat the DDPG paper with both parallel and traditional approaches to check if the parallel one has real advantages. Do the same for different types of noise.
- Build a nice code, that supports incremental learning.
