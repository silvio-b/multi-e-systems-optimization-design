import torch as t
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
from agents.buffer import ReplayBuffer
from networks.networks import SoftQNetworkDiscrete, PolicyNetworkDiscrete


class SACAgent():
    def __init__(self, state_dim=None,
                 action_dim=None, hidden_dim=None, discount=0.99, tau=0.005, lr_actor=None, lr_critic=None, batch_size=256,
                 replay_buffer_capacity=1e5, learning_start=None, reward_scaling=1., seed=0, rbc_controller=None,
                 safe_exploration=None, automatic_entropy_tuning=False, alpha=1):

        if hidden_dim is None:
            hidden_dim = [256, 256]
        self.learning_start = learning_start
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        self.reward_scaling = reward_scaling
        t.manual_seed(seed)
        np.random.seed(seed)
        self.action_list_ = []
        self.action_list2_ = []
        self.hidden_dim = hidden_dim
        self.rbc_controller = rbc_controller
        self.safe_exploration = safe_exploration
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.time_step = 0

        # Optimizers/Loss using the Huber loss
        # self.soft_q_criterion = f.mse_loss

        # device
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        self.memory = ReplayBuffer(input_shape=int(state_dim), n_actions=int(1),
                                   max_mem_size=int(replay_buffer_capacity))

        # init networks
        self.soft_q_net1 = SoftQNetworkDiscrete(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetworkDiscrete(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_soft_q_net1 = SoftQNetworkDiscrete(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_soft_q_net2 = SoftQNetworkDiscrete(state_dim, action_dim, hidden_dim).to(self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(),
                                       self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_soft_q_net2.parameters(),
                                       self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        # Policy
        self.policy_net = PolicyNetworkDiscrete(state_dim, action_dim, [64, 64]).to(self.device)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr_critic)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr_critic)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)

        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / action_dim)) * 0.98
            self.log_alpha = t.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_critic, eps=1e-4)
        else:
            self.alpha = alpha

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

    def get_actions_probabilities(self, observation):

        if self.device.type == "cuda":
            state = t.cuda.FloatTensor([observation]).to(self.device)
        else:
            state = t.FloatTensor([observation]).to(self.device)
        _, (actions_probabilities, _), _ = self.policy_net.sample(state)

        return actions_probabilities.cpu().detach().numpy()[0]

    def get_q_values(self, observation):

        if self.device.type == "cuda":
            state = t.cuda.FloatTensor([observation]).to(self.device)
        else:
            state = t.FloatTensor([observation]).to(self.device)
        q_1 = self.soft_q_net1(state)
        q_2 = self.soft_q_net2(state)

        q_1 = q_1.cpu().detach().numpy()[0]
        q_2 = q_2.cpu().detach().numpy()[0]

        return q_1, q_2

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        if self.device.type == "cuda":
            state = t.cuda.FloatTensor(state).to(self.device)
            next_state = t.cuda.FloatTensor(next_state).to(self.device)
            action = t.cuda.LongTensor(action).to(self.device)
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

            new_next_actions, (action_probabilities, log_action_probabilities), _ = self.policy_net.sample(next_state)

            qf1_next_target = self.target_soft_q_net1(next_state)
            qf2_next_target = self.target_soft_q_net2(next_state)

            min_qf_next_target = action_probabilities * (
                        t.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)

            q_target = reward + (1 - done) * self.discount * min_qf_next_target
            # self.q_tracker.append(q_target.mean())

        # Update Soft Q-Networks
        q1_pred = self.soft_q_net1(state)
        q2_pred = self.soft_q_net2(state)

        q1_pred = q1_pred.gather(1, action.reshape([self.batch_size, 1]))
        q2_pred = q2_pred.gather(1, action.reshape([self.batch_size, 1]))

        q1_loss = f.mse_loss(q1_pred, q_target)
        q2_loss = f.mse_loss(q2_pred, q_target)

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()

        # Update Policy
        new_actions, (action_probabilities, log_action_probabilities), _ = self.policy_net.sample(state)

        min_qf_pi = t.min(
            self.soft_q_net1(state),
            self.soft_q_net2(state)
        )

        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = t.sum(log_action_probabilities * action_probabilities, dim=1)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        else:
            alpha_loss = None

        if alpha_loss is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

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

    def save_models(self, path):
        print('...saving models...')
        t.save(self.soft_q_net1, path + '\\critic_1.pth')
        t.save(self.soft_q_net2, path + '\\critic_2.pth')
        t.save(self.policy_net, path + '\\actor.pth')

    def load_models(self, path):
        print('...loading models...')
        dev = self.device
        self.soft_q_net1 = t.load(path + '\\critic_1.pth', map_location=dev)
        self.soft_q_net2 = t.load(path + '\\critic_2.pth', map_location=dev)
        self.policy_net = t.load(path + '\\actor.pth', map_location=dev)
