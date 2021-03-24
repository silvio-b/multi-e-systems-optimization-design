import numpy as np
import math
import random


class ReplayBuffer:
    def __init__(self, input_shape, n_actions, max_mem_size=1000000, n_batch=None):
        self.mem_size = max_mem_size
        self.mem_ctr = 0
        self.n_batch = n_batch
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)  # How many slots we have already stored

        batch = np.random.choice(max_mem, batch_size)  # List of random integers between 0 and
        # max memory of length batch size

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


    def sample_buffer_online(self, batch_size):
        # print('Not implemented')
        max_mem = min(self.mem_ctr, self.mem_size)  # How many slots we have already stored

        #Creo la lista con gli indici in modo da poter poi eliminare da qui quelli che sono presi del batch


        #Qui dovrebbe impostarsi un ciclo for sulla dimensione di n_batch in modo da prendere i batch ed eliminare dalla lista gli indici già scelti
        n_batch = math.floor(max_mem / batch_size)

        batch_indexes = list(range(0, max_mem))

        states = np.zeros((n_batch, batch_size, self.input_shape))
        new_states = np.zeros((n_batch, batch_size, self.input_shape))
        actions = np.zeros((n_batch, batch_size, self.n_actions))
        rewards = np.zeros((n_batch, batch_size))
        dones = np.zeros((n_batch, batch_size), dtype=bool)

        for index in range(0, n_batch):

            batch = random.sample(batch_indexes, batch_size)  # --> In questo modo li prende solo una volta e senza ripetere lo stesso indice

            states[index] = self.state_memory[batch]
            new_states[index] = self.new_state_memory[batch]
            actions[index] = self.action_memory[batch]
            rewards[index] = self.reward_memory[batch]
            dones[index] = self.terminal_memory[batch]

            # delete from the list of possible index already selected
            batch_indexes = [element for element in batch_indexes if element not in batch]

        return states, actions, rewards, new_states, dones
