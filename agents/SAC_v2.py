import torch as t
import torch.optim as optim
import torch.nn as nn
import numpy as np
from agents.buffer import ReplayBuffer
from networks.networks import SoftQNetwork, PolicyNetwork


class SAC2Agent():
    def __init__(self, observation_space=None,
                 action_space=None, hidden_dim=None, discount=0.99, tau=0.005, lr=None, batch_size=256,
                 replay_buffer_capacity=1e5, start_training=None,
                 exploration_period=None, action_scaling_coef=1., reward_scaling=1., update_per_step=1, iterations_as=2,
                 seed=0, deterministic=None, rbc_controller=None, safe_exploration=None):

        if hidden_dim is None:
            hidden_dim = [256, 256]
        self.start_training = start_training
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.action_scaling_coef = action_scaling_coef
        self.reward_scaling = reward_scaling
        t.manual_seed(seed)
        np.random.seed(seed)
        self.deterministic = deterministic
        self.update_per_step = update_per_step
        self.iterations_as = iterations_as
        self.exploration_period = exploration_period
        self.action_list_ = []
        self.action_list2_ = []
        self.hidden_dim = hidden_dim
        self.rbc_controller = rbc_controller
        self.safe_exploration = safe_exploration
        self.reset_action_tracker()

        self.reset_reward_tracker()

        self.time_step = 0
        self.action_space = action_space
        self.observation_space = observation_space

        # Optimizers/Loss using the Huber loss
        self.soft_q_criterion = nn.SmoothL1Loss()

        # device
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        self.alpha = 0.05

        self.memory = ReplayBuffer(input_shape=int(state_dim), n_actions=int(action_dim),
                                   max_mem_size=int(replay_buffer_capacity))

        # init networks
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        # Policy
        self.policy_net = PolicyNetwork(state_dim, action_dim, self.action_space, self.action_scaling_coef,
                                        hidden_dim).to(self.device)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_entropy = -np.prod(self.action_space.shape).item()
        self.log_alpha = t.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def reset_action_tracker(self):
        self.action_tracker = []

    def reset_reward_tracker(self):
        self.reward_tracker = []

    def choose_action(self, simulation_step, electricity_price, storage_soc, observation):

        if simulation_step < self.safe_exploration:
            action = self.rbc_controller.choose_action(electricity_price=electricity_price,
                                                       storage_soc=storage_soc)
            actions = t.tensor([action], dtype=t.float).to(self.device)
            # print(action)
        else:
            if self.device.type == "cuda":
                state = t.cuda.FloatTensor([observation]).to(self.device)
            else:
                state = t.FloatTensor([observation]).to(self.device)
            actions, _, _ = self.policy_net.sample(state)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        if self.device.type == "cuda":
            state = t.cuda.FloatTensor(state).to(self.device)
            next_state = t.cuda.FloatTensor(next_state).to(self.device)
            action = t.cuda.FloatTensor(action).to(self.device)
            reward = t.cuda.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = t.cuda.FloatTensor(done).unsqueeze(1).to(self.device)
        else:
            state = t.FloatTensor(state).to(self.device)
            next_state = t.FloatTensor(next_state).to(self.device)
            action = t.FloatTensor(action).to(self.device)
            reward = t.FloatTensor(reward).unsqueeze(1).to(self.device)
            done = t.FloatTensor(done).unsqueeze(1).to(self.device)

        with t.no_grad():
            # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) state and its associated log probability of occurrence.
            new_next_actions, new_log_pi, _ = self.policy_net.sample(next_state)

            target_q_values = t.min(
                self.target_soft_q_net1(next_state, new_next_actions),
                self.target_soft_q_net2(next_state, new_next_actions),
            ) - self.alpha * new_log_pi

            q_target = reward + (1 - done) * self.discount * target_q_values
            # self.q_tracker.append(q_target.mean())

        # Update Soft Q-Networks
        q1_pred = self.soft_q_net1(state, action)
        q2_pred = self.soft_q_net2(state, action)

        q1_loss = self.soft_q_criterion(q1_pred, q_target)
        q2_loss = self.soft_q_criterion(q2_pred, q_target)

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        # Update Policy
        new_actions, log_pi, _ = self.policy_net.sample(state)

        q_new_actions = t.min(
            self.soft_q_net1(state, new_actions),
            self.soft_q_net2(state, new_actions)
        )

        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.alpha = 0.05  # self.log_alpha.exp()

        # Soft Updates
        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    # def save_models(self, path, name):
    #     for uid in self.building_ids:
    #         torch.save(self.soft_q_net1[uid].state_dict(), os.path.join(path, uid + name + 'soft_q_net1.pth'))
    #         torch.save(self.soft_q_net2[uid].state_dict(), os.path.join(path, uid + name + 'soft_q_net2.pth'))
    #         torch.save(self.target_soft_q_net1[uid].state_dict(),
    #                    os.path.join(path, uid + name + 'target_soft_q_net1.pth'))
    #         torch.save(self.target_soft_q_net2[uid].state_dict(),
    #                    os.path.join(path, uid + name + 'target_soft_q_net2.pth'))
    #         torch.save(self.policy_net[uid].state_dict(), os.path.join(path, uid + name + 'policy_net.pth'))
    #
    # def load_models(self, uid, path, name):
    #
    #     state_dim = self.observation_spaces[uid].shape[0]
    #     action_dim = self.action_spaces[uid].shape[0]
    #
    #     model1 = SoftQNetwork(state_dim, action_dim, self.hidden_dim)
    #     model2 = SoftQNetwork(state_dim, action_dim, self.hidden_dim)
    #     model3 = SoftQNetwork(state_dim, action_dim, self.hidden_dim)
    #     model4 = SoftQNetwork(state_dim, action_dim, self.hidden_dim)
    #     model5 = PolicyNetwork(state_dim, action_dim, self.action_spaces[uid], self.action_scaling_coef,
    #                            self.hidden_dim)
    #     model1.load_state_dict(torch.load(os.path.join(path, uid + name + 'soft_q_net1.pth')))
    #     model2.load_state_dict(torch.load(os.path.join(path, uid + name + 'soft_q_net2.pth')))
    #     model3.load_state_dict(torch.load(os.path.join(path, uid + name + 'target_soft_q_net1.pth')))
    #     model4.load_state_dict(torch.load(os.path.join(path, uid + name + 'target_soft_q_net2.pth')))
    #     model5.load_state_dict(torch.load(os.path.join(path, uid + name + 'policy_net.pth')))
    #
    #     return model1, model2, model3, model4, model5