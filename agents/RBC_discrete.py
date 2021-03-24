import numpy as np
import torch as t
from torch.distributions.categorical import Categorical


class RBCAgent:
    def __init__(self, min_storage_soc, min_charging_storage_soc, max_storage_soc, min_electricity_price,
                 random_choice=False, random_probability=0.1):
        self.min_storage_soc = min_storage_soc
        self.min_charging_storage_soc = min_charging_storage_soc
        self.max_storage_soc = max_storage_soc
        self.min_electricity_price = min_electricity_price
        self.charge_flag = 0
        self.random_choice = random_choice
        self.random_probability = random_probability

    def choose_action(self, electricity_price, storage_soc):
        if electricity_price == self.min_electricity_price:
            if self.charge_flag == 0 and storage_soc < self.min_charging_storage_soc:
                action = 1  # Charge the storage at nominal mass flow rate
                self.charge_flag = 1
            elif self.charge_flag == 1 and storage_soc < self.max_storage_soc:
                action = 1  # Keep charging the storage at nominal mass flow rate
                self.charge_flag = 1
            elif self.charge_flag == 1 and storage_soc >= self.max_storage_soc:
                action = 0  # the storage is full. stop the charging phase
                self.charge_flag = 0
            else:
                action = 0  # the storage is full. stop the charging phase
                self.charge_flag = 0
        else:
            # in this condition the price of electricity is not low. It is convinient to discharge the stored energy
            self.charge_flag = 0
            if storage_soc > self.min_storage_soc:
                action = 0  # DisCharge the storage at nominal mass flow rate
            else:
                action = 0  # the storage is empty. stop the charging phase

        if self.random_choice:
            if np.random.uniform() < self.random_probability:
                action = np.random.randint(-1, 2)

        action = np.array([action])

        return action

    def sample(self, action, device):

        action_probabilities = t.zeros((action.shape[0], 2), device=device)
        if device.type == "cuda":
            action_probabilities = t.cuda.FloatTensor(action_probabilities).to(device)
        else:
            action_probabilities = t.FloatTensor(action_probabilities).to(device)

        for i in range(0, action_probabilities.shape[0]):
            action_probabilities[i][action[i]] = 1.

        max_probability_action = t.argmax(action_probabilities, dim=-1)
        action_distribution = Categorical(action_probabilities)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = t.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action







