import numpy as np
from collections import deque
from ReplayBuffer import ReplayBuffer
from networks import *
import sys
import tensorflow as tf
import gym


TRAIN_START = 10_000
BUFFER_LENGTH = 200_000
FINAL_EXPLORATION_FRAME = 100_000
MIN_EPS = 0.02
STEPS_PER_NETWORK_UPDATE = 4
DISCOUNT_FACTOR = 0.99
STEPS_PER_TARGET_UPDATE = 1_000
MINIBATCH_SIZE = 64
TRAINING_STEPS = 1_000_000
BETA_ANNEAL_STEPS = TRAINING_STEPS
LOG_STEPS = 1_000
LOG_FILE = "log.txt"


class DQNAgent:
    def __init__(self, env, model_key,
                 use_double_dqn=False, use_dueling_dqn=False, use_priority=False,
                 normalize_reward_coeff=1):
        self.env = env
        self.total_steps_taken = 0
        self.steps_since_last_Q_update = 0
        self.steps_since_last_Target_update = 0
        self.steps_since_last_log = 0
        self.n_q_updates = 0
        self.n_target_updates = 0
        self.n_episodes = 0
        self.best_mean_reward = -np.inf
        self.use_double_dqn = use_double_dqn
        self.use_priority = use_priority
        if self.use_priority:
            self.beta0 = 0.4
            self.beta_anneal = (1 - self.beta0)/TRAINING_STEPS
            self.replay_buffer = ReplayBuffer(state_shape=env.observation_space.shape,
                                              action_shape=1,
                                              buffer_length=BUFFER_LENGTH,
                                              use_priority=self.use_priority,
                                              alpha=0.6,
                                              normalize_reward_coeff=normalize_reward_coeff)
        else:
            self.replay_buffer = ReplayBuffer(state_shape=env.observation_space.shape,
                                              action_shape=1,
                                              buffer_length=BUFFER_LENGTH,
                                              use_priority=self.use_priority,
                                              normalize_reward_coeff=normalize_reward_coeff)

        self.network_loss = deque(maxlen=1000)
        self.rewards = deque(maxlen=1000)
        self.q_network = FeedForwardPolicy(env.observation_space.shape,
                                           env.action_space.n,
                                           "trained_models/q_network_{}/model.ckpt".format(model_key),
                                           dueling=use_dueling_dqn,
                                           scope="q_network")
        self.target_network = FeedForwardPolicy(env.observation_space.shape,
                                                env.action_space.n,
                                                "trained_models/target_network_{}/model.ckpt".format(model_key),
                                                dueling=use_dueling_dqn,
                                                scope="target_network")
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q_network")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_network")
        self.update_target_ops = []
        for from_var, to_var in zip(q_vars, target_vars):
            self.update_target_ops.append(to_var.assign(from_var))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.q_network.set_session(self.sess)
        self.target_network.set_session(self.sess)
        self.update_target_network()

        '''
        state = np.array([[-2, 1, 14, 10],
                          [10, 15, -3, 2]])
        print(state)
        self.q_network.predict(state)
        print("*****")
        self.q_network.predict(state[0])
        print("*****")
        self.q_network.predict(state[1])
        exit()
        '''

    def update_target_network(self):
        self.sess.run(self.update_target_ops)
        self.n_target_updates += 1

    def learn(self):
        while True:
            s = self.env.reset()
            episode_reward = 0

            while True:
                self.steps_since_last_log += 1

                if self.total_steps_taken > TRAIN_START:
                    eps = (1 - MIN_EPS) * (self.total_steps_taken/FINAL_EXPLORATION_FRAME)
                else:
                    eps = 1
                if np.random.random() > eps:
                    s_copy = np.copy(s)
                    a = np.argmax(self.q_network.predict(s_copy))
                else:
                    a = self.env.action_space.sample()
                s_, r, t, _ = self.env.step(a)
                episode_reward += r
                self.replay_buffer.append([s, a, r, s_, t])
                self.total_steps_taken += 1
                s = s_
                if self.total_steps_taken > TRAIN_START:
                    self.steps_since_last_Q_update += 1
                    if self.steps_since_last_Q_update >= STEPS_PER_NETWORK_UPDATE:
                        self.steps_since_last_Q_update = 0
                        self.train_q_network()
                        self.steps_since_last_Target_update += 1
                        if self.steps_since_last_Target_update >= STEPS_PER_TARGET_UPDATE:
                            self.update_target_network()
                            self.steps_since_last_Target_update = 0
                if t:
                    self.rewards.append(episode_reward)
                    self.n_episodes += 1
                    break
            if self.total_steps_taken > TRAINING_STEPS:
                self.update_target_network()
                self.write_log()
                break
            if self.steps_since_last_log >= LOG_STEPS:
                self.steps_since_last_log = 0
                self.write_log()

    def train_q_network(self):
        if self.use_priority:
            sample_idx, weights, states, actions, rewards, next_states, terminals = \
                self.replay_buffer.sample(MINIBATCH_SIZE, beta=self.beta0)
            self.beta0 += self.beta_anneal
            updated_priorities = []
        else:
            states, actions, rewards, next_states, terminals = \
                self.replay_buffer.sample(MINIBATCH_SIZE)
            weights = np.ones_like(terminals)
        outputs = []

        # for s, a, r, s_, t in zip(states, actions, rewards, next_states, terminals):
        for i in range(len(terminals)):
            s = states[i]
            r = rewards[i]
            a = actions[i]
            s_ = next_states[i]
            t = terminals[i]
            target = 0
            if t:
                target = r
            else:
                if self.use_double_dqn:
                    best_action = np.argmax(self.q_network.predict(s_))
                    target = r + DISCOUNT_FACTOR * self.target_network.predict(s_)[0][best_action]
                else:
                    target = r + DISCOUNT_FACTOR * np.max(self.target_network.predict(s_))
            outputs.append(target)
            if self.use_priority:
                updated_priorities.append(np.abs(target - self.q_network.predict(s)[0][a]))
        if self.use_priority:
            self.replay_buffer.update_priorities(sample_idx, updated_priorities)
        outputs = np.array(outputs)
        loss = self.q_network.train_step(states, actions, outputs, weights)
        self.network_loss.append(loss)
        self.n_q_updates += 1

    def write_log(self):
        mean_rewards = np.mean(self.rewards)
        eval_rewards = self.test_policy()
        mean_network_loss = np.mean(self.network_loss)
        text = "Num Steps: {}, Num Episodes: {}, Mean Rewards: {:.2f}, Evaluation Rewards: {:.2f}, Mean Network Loss: {:.2f}, Q Updates: {}, Target Updates: {}, Best Model: {:.2f}".format(
             self.total_steps_taken, self.n_episodes, mean_rewards, eval_rewards, mean_network_loss, self.n_q_updates, self.n_target_updates, self.best_mean_reward)
        print(text)
        with open(LOG_FILE, 'a') as f:
            f.write("{}\n".format(text))
        if eval_rewards > self.best_mean_reward:
            self.best_mean_reward = eval_rewards
            self.q_network.save()
        self.target_network.save()

    def load_model(self, model_name):
        self.q_network.load(model_name)

    def test_policy(self, render=False):
        # self.target_network.load()
        rewards = []
        for _ in range(10):
            s = self.env.reset()
            episode_reward = 0
            while True:
                s_copy = np.copy(s)
                a = np.argmax(self.q_network.predict(s_copy))
                s_, r, t, _ = self.env.step(a)
                episode_reward += r
                s = s_
                if render:
                    self.env.render()
                if t:
                    break
            rewards.append(episode_reward)
        return np.mean(rewards)


if __name__ == "__main__":
    mode = sys.argv[1]
    model_key = "lander"
    env = gym.make("LunarLander-v2")
    agent = DQNAgent(env, model_key=model_key, use_dueling_dqn=True, use_priority=True,
                     normalize_reward_coeff=10)
    if mode == "test":
        agent.load_model("trained_models/q_network_{}/model.ckpt".format(model_key))
        agent.test_policy(render=True)
    else:
        agent.learn()
