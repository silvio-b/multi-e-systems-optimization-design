import torch as t
import torch.nn.functional as f
from agents.buffer import ReplayBuffer
from networks.ActorNetwork import ActorNetwork
from networks.CriticNetwork import CriticNetwork
from networks.ValueNetwork import ValueNetwork
from agents.RBC import RBCAgent


class SACAgent:
    def __init__(self, lr_actor=0.0003, lr_critic=0.0003, state_dim=8, discount=0.99,
                 action_dim=1, replay_buffer_capacity=1000000, tau=0.005, batch_size=256, reward_scaling=1,
                 rbc_controller=RBCAgent,
                 safe_exploration=None, hidden_dim=None):
        self.gamma = discount
        self.tau = tau
        self.memory = ReplayBuffer(input_shape=state_dim, n_actions=action_dim, max_mem_size=replay_buffer_capacity)
        self.batch_size = batch_size
        self.n_actions = action_dim
        self.rbc_controller = rbc_controller
        self.safe_exploration = safe_exploration
        self.hidden_size = hidden_dim

        self.actor = ActorNetwork(learning_rate=lr_actor, input_size=state_dim, max_action=1, n_actions=action_dim,
                                  name='actor', hidden_size=self.hidden_size)
        self.critic_1 = CriticNetwork(learning_rate=lr_critic, input_size=state_dim, n_actions=action_dim,
                                      name='critic_1', hidden_size=self.hidden_size)
        self.critic_2 = CriticNetwork(learning_rate=lr_critic, input_size=state_dim, n_actions=action_dim,
                                      name='critic_2', hidden_size=self.hidden_size)

        self.value = ValueNetwork(learning_rate=lr_critic, input_size=state_dim, name='value',
                                  hidden_size=self.hidden_size)
        self.target_value = ValueNetwork(learning_rate=lr_critic, input_size=state_dim, name='target_value',
                                         hidden_size=self.hidden_size)

        self.scale = reward_scaling
        self.update_network_parameters(tau=1)

    def choose_action(self, simulation_step, electricity_price, storage_soc, observation):

        if simulation_step < self.safe_exploration:
            action = self.rbc_controller.choose_action(electricity_price=electricity_price,
                                                       storage_soc=storage_soc)
            actions = t.tensor([action], dtype=t.float).to(self.actor.device)
            # print(action)
        else:
            state = t.tensor([observation], dtype=t.float).to(self.actor.device)
            actions, _ = self.actor.sample_normal(state, rep=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self, path):
        print('...saving models...')
        t.save(self.actor, path + '\\actor.pth')
        t.save(self.value, path + '\\value.pth')
        t.save(self.target_value, path + '\\target_value.pth')
        t.save(self.critic_1, path + '\\critic_1.pth')
        t.save(self.critic_2, path + '\\critic_2.pth')

    def load_models(self, path):
        print('...loading models...')
        dev = self.actor.device
        self.actor = t.load(path + '\\actor.pth', map_location=dev)
        self.value = t.load(path + '\\value.pth', map_location=dev)
        self.target_value = t.load(path + '\\target_value.pth', map_location=dev)
        self.critic_1 = t.load(path + '\\critic_1.pth', map_location=dev)
        self.critic_2 = t.load(path + '\\critic_2.pth', map_location=dev)

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = t.tensor(reward, dtype=t.float).to(self.actor.device)
        done = t.tensor(done).to(self.actor.device)
        new_state = t.tensor(new_state, dtype=t.float).to(self.actor.device)
        state = t.tensor(state, dtype=t.float).to(self.actor.device)
        action = t.tensor(action, dtype=t.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(new_state).view(-1)
        value_[done] = 0.0

        actions, log_prob = self.actor.sample_normal(state, rep=False)
        log_prob = log_prob.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = t.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.opt.zero_grad()
        value_target = critic_value - log_prob
        value_loss = 0.5 * f.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.opt.step()

        actions, log_prob = self.actor.sample_normal(state, rep=True)
        log_prob = log_prob.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = t.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_prob - critic_value
        actor_loss = t.mean(actor_loss)
        self.actor.opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.opt.step()

        self.critic_1.opt.zero_grad()
        self.critic_2.opt.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * f.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * f.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.opt.step()
        self.critic_2.opt.step()

        self.update_network_parameters()

    def learn_actor(self, updates: int, batch_size):

        for i in range(0, updates):
            print(i)
            state, _, _, _, _ = self.memory.sample_buffer(batch_size)

            state = t.tensor(state, dtype=t.float).to(self.actor.device)

            actions, log_prob = self.actor.sample_normal(state, rep=True)
            log_prob = log_prob.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = t.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)

            actor_loss = log_prob - critic_value
            actor_loss = t.mean(actor_loss)
            self.actor.opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.opt.step()