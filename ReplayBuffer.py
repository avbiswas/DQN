import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, state_shape, action_shape, use_priority=False,
                 buffer_length=1_000_000, alpha=0.6, normalize_reward_coeff=1):
        # self.replay_buffer = deque(maxlen=buffer_length)
        self.buffer_length = buffer_length
        self.use_priority = use_priority
        self.normalize_reward_coeff = normalize_reward_coeff
        self.states_buffer = np.zeros([self.buffer_length, *state_shape])
        self.action_buffer = np.zeros([self.buffer_length, ], np.int8)
        self.reward_buffer = np.zeros([self.buffer_length, ])
        self.next_states_buffer = np.zeros([self.buffer_length, *state_shape])
        self.terminal_buffer = np.zeros([self.buffer_length, ], np.int8)
        self.alpha = alpha
        if self.use_priority:
            self.priorities = np.zeros([self.buffer_length, ], np.float16)
        self.current_counter = 0
        self.buffer_full = False

    def update_priorities(self, sample_idx, update_priorities):
        self.priorities[sample_idx] = np.array(update_priorities) ** self.alpha
        # print(self.priorities)

    def append(self, observation):
        # self.replay_buffer.append(observation)
        s, a, r, s_, t = observation
        self.states_buffer[self.current_counter] = s
        self.action_buffer[self.current_counter] = a
        self.reward_buffer[self.current_counter] = r/self.normalize_reward_coeff
        self.next_states_buffer[self.current_counter] = s_
        self.terminal_buffer[self.current_counter] = t
        if self.use_priority:
            self.priorities[self.current_counter] = 1

        if self.current_counter + 1 >= self.buffer_length:
            self.buffer_full = True

        self.current_counter = (self.current_counter + 1) % self.buffer_length

    def sample(self, n_rows=32, beta=0.5):
        length = self.buffer_length if self.buffer_full else self.current_counter
        if self.use_priority:
            priorities = self.priorities[:length]
            priorities = priorities/np.sum(priorities)
            random_samples = np.random.choice(length, n_rows, p=priorities)
            priorities = self.priorities[random_samples]
            weights = np.power((length * priorities), -beta)
            weights = weights/np.max(weights)
        else:
            random_samples = np.random.choice(length, n_rows)
        # print(np.max(self.reward_buffer[random_samples]))
        if self.use_priority:
            return random_samples, \
                   weights, \
                   self.states_buffer[random_samples], \
                   self.action_buffer[random_samples], \
                   self.reward_buffer[random_samples], \
                   self.next_states_buffer[random_samples], \
                   self.terminal_buffer[random_samples]
        else:
            return self.states_buffer[random_samples], \
                   self.action_buffer[random_samples], \
                   self.reward_buffer[random_samples], \
                   self.next_states_buffer[random_samples], \
                   self.terminal_buffer[random_samples]
