# Lunar Lander-V2 using DQN and Actor Critic Methods

## Setting up project environment:
- Create new enviroment using anaconda or miniconda
```
conda create -n env_name pthon=3.6
```
- Install dependencies from requirements.txt
```
pip install -r requirements.txt
```

After installing dependencies, [DQN](DQN) and [Actor Critic](ActorCritic) implemented in respective folders.

### DQN:
* DQN algorithm is implemented in [DQN.py](DQN/DQN.py)
* For training and testing the DQN, use the [dqn_run.ipynb](DQN/dqn_run.ipynb)
* DQN testing results was stored in [DQN results](DQN/dqn_results) folder
* DQN trained model was stored in [DQN model](DQN/saved_networks) folder

### Actor Critic:
* Actor critic algorithm is implemented in [main.py](ActorCritic/main.py)
* For training and testing the Actor Critic use the [actor_critic_run.ipynb](ActorCritic/actor_critic_run.ipynb) folder
* Actor Critic testing results was stored in [actor critic results](ActorCritic/actor_critic_results) folder.
* Actor critic model is stored in [Actor Critic](ActorCritic/actor_crtic_model)
