from abc import ABC

import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from tensorflow.keras import layers


class ActorCritic(tf.keras.Model, ABC):
    def __init__(self,num_actions, num_hidden_units):
        super().__init__()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.hidden_layer = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        input_layer = self.common(inputs)
        layer1 = self.hidden_layer(input_layer)
        # layer2 = self.hidden_layer(layer1)
        # x = self.common(inputs)
        return self.actor(input_layer), self.critic(input_layer)


class Agent:
    def __init__(self, env, gamma=0.99, learning_rate=0.01, num_hidden_units=128):
        self.env = env
        self.num_actions = env.action_space.n  # 4
        self.num_hidden_units = num_hidden_units
        # Create Actor critic base model
        self.model = ActorCritic(self.num_actions, self.num_hidden_units)
        self.max_steps_per_episode = 1000

        # Discount factor for future rewards
        self.gamma = gamma

        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

    def env_step(self, action):
        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action):
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])

    def run_episode(self, initial_state, model, max_steps):
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            rewards = rewards.write(t, reward)

            # Terminate if episode is completed
            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(self, rewards, standardize: bool = True):

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + self.eps))

        return returns

    def compute_loss(self, action_probs, values, returns) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)
        # Return actor + critic loss
        return actor_loss + critic_loss

    @tf.function
    def train_step(self, initial_state):
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(
                initial_state, self.model, self.max_steps_per_episode)

            # Calculate expected returns
            returns = self.get_expected_return(rewards)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        _episode_reward = tf.math.reduce_sum(rewards)

        return _episode_reward


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    # Set seed for experiment reproducibility
    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    min_episodes_criterion = 100
    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    reward_threshold = 200
    running_reward = 0
    episodes_reward = []
    agent = Agent(env)
    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(agent.env.reset(), dtype=tf.float32)
            episode_reward = int(agent.train_step(initial_state))

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)
            average_reward = 0
            if len(episodes_reward) > 100:
                average_reward = np.mean(episodes_reward[-100:])

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, average_reward=average_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if average_reward > reward_threshold and i >= min_episodes_criterion:
                break

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
