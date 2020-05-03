
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

This is my implementation of the second project from  [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). For this project, I was working with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.04 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Solving the Environment

The task is episodic, and in order to solve the environment,  the agent must get an average score of +30 over 100 consecutive episodes.


### Getting Started

1. Download and install the [drlnd repository](https://github.com/udacity/deep-reinforcement-learning). The installation instruction is in their readme file.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **One (1) Agent Version**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

3. Place the file in the DRLND GitHub repository, in the folder of this repository, and unzip (or decompress) the file. 

### Instructions

#### Running training from scratch
Run the `Countinus_Control.ipynb` notebook to train your agent. Agent in the notebook is being trained for 2,000 episodes and there is no need to train the agent for such a long time. 1,000 episodes is probably enough. You can change the number of episodes in `generate_episodes` function.
*Note - the speed of learning is quite random, so do not be surprised if you cannot fully repeat my experiment*.

#### Testing models
After training your models run the `TestAgents.ipynb` notebook to test the agents without noise. This notebook makes each agent to control the robot arm for 100 episodes and collects the results.

#### Analyzing results
Run the `AnalyzeResults.ipynb` notebook, to analyze the results. This notebook will:
    - analyze the results and displays charts with summary
    - load a trained agent and visualize how it works
    
### Credits
I used the [Udacity DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) implementation as a base for my code.
