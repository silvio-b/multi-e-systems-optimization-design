import os
from GymEnvironments.environment_continuous_action import RelicEnv
import pandas as pd
from agents.SAC import SACAgent
from utils import order_state_variables, min_max_scaling, calculate_tank_soc
import numpy as np

directory = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':

    result_directory_path = 'D:\\OneDrive - Politecnico di Torino\\PhD_Silvio\\14_Projects\\002_PVZenControl\\Thermal_Electrical_Storage_Control\\'
    result_directory = 'test_02'
    safe_exploration = -1
    discount_factor = 0.99
    alpha = 0.25
    tau = 0.005
    automatic_entropy_tuning = False
    learning_rate_actor = 0.005
    learning_rate_critic = 0.005
    n_hidden_layers = 2
    n_neurons = 256
    batch_size = 256
    replay_buffer_capacity = 24 * 30 * 100
    prediction_observations = ['electricity_price', 'cooling_load', 'pv_power_generation']
    prediction_horizon = 24
    min_temperature_limit = 10  # Below this value no charging
    min_charging_temperature = 12  # Charging begins above this threshold
    max_temperature_limit = 18  # Above this threshold no discharging
    num_episodes = 10

    result_directory_final = result_directory_path + result_directory
    if not os.path.exists(result_directory_final):
        os.makedirs(result_directory_final)

    hidden_size = n_hidden_layers * [n_neurons]

    config = {
        'res_directory': result_directory_final,
        # Change this folder to the path where you want to save the output of each episode
        'weather_file': 'ITA_TORINO-CASELLE_IGDG',
        'simulation_days': 90,
        'tank_min_temperature': min_temperature_limit,
        'tank_max_temperature': max_temperature_limit,
        'price_schedule_name': 'electricity_price_schedule.csv'}

    env = RelicEnv(config)

    # Import predictions
    cooling_load_predictions = pd.read_csv('supportFiles\\prediction-cooling_load_perfect.csv')
    electricity_price_predictions = pd.read_csv('supportFiles\\prediction-electricity_price_perfect.csv')
    pv_power_generation_predictions = pd.read_csv('supportFiles\\prediction-pv_power_generation_perfect.csv')
    electricity_price_schedule = pd.read_csv('supportFiles\\electricity_price_schedule.csv', header=None)

    # Set the number of actions
    n_actions = 2
    input_dims = env.observation_space.shape[0]

    # define period for RBC control and
    min_price = float(electricity_price_schedule[0].min())
    min_temperature_limit = 10  # Below this value no charging
    min_charging_temperature = 12  # Charging begins above this threshold
    max_temperature_limit = 18  # Above this threshold no discharging

    # evaluate SOC
    max_storage_soc = calculate_tank_soc(min_temperature_limit, min_temperature_limit,
                                         max_temperature_limit)  # The storage is full
    min_charging_storage_soc = calculate_tank_soc(min_charging_temperature, min_temperature_limit,
                                                  max_temperature_limit)
    min_storage_soc = calculate_tank_soc(max_temperature_limit, min_temperature_limit,
                                         max_temperature_limit)  # The storage is empty

    rbc_controller = None
    # Initialize agent
    agent = SACAgent(state_dim=input_dims,
                     action_dim=n_actions, hidden_dim=hidden_size, discount=discount_factor, tau=tau,
                     lr_critic=learning_rate_critic, lr_actor=learning_rate_actor,
                     batch_size=batch_size, replay_buffer_capacity=replay_buffer_capacity,
                     reward_scaling=10., rbc_controller=rbc_controller, safe_exploration=safe_exploration)

    # Define the number of episodes
    score_history = []

    # Training Loop
    for episode in range(1, num_episodes + 1):
        episode_step = 0
        observation = env.reset()
        # append prediction
        electricity_price = electricity_price_schedule[0][env.kStep + 1]
        storage_soc = observation[3]
        observation = order_state_variables(env=env,
                                            observation=observation,
                                            cooling_load_predictions=cooling_load_predictions,
                                            electricity_price_predictions=electricity_price_predictions,
                                            pv_power_generation_predictions=pv_power_generation_predictions,
                                            horizon=24,
                                            step=episode_step)
        # Scale observations
        observation = min_max_scaling(observation, env.state_mins, env.state_maxs, np.array([0]),
                                      np.array([1]))

        score = 0
        done = False
        actions_probabilities = []
        q_values_1 = []
        q_values_2 = []

        while not done:

            action = agent.choose_action(simulation_step=env.kStep + (episode - 1) * 10000,
                                         electricity_price=electricity_price,
                                         storage_soc=storage_soc,
                                         observation=observation)

            step = 1
            reward = 0
            cooling_load = 0  # the cooling load needs to be averaged across simulation steps
            auxiliary_load = 0
            pv_power = 0
            while step <= env.ep_time_step:
                new_observation, reward_step, done, info = env.step(action)
                cooling_load += new_observation[1]
                auxiliary_load += new_observation[9]
                pv_power += new_observation[8]
                reward += reward_step
                step += 1

            cooling_load = cooling_load / env.ep_time_step  # calculate average value
            pv_power = pv_power / env.ep_time_step
            auxiliary_load = auxiliary_load / env.ep_time_step
            episode_step += 1

            if done:
                break

            # append predictions
            electricity_price = electricity_price_schedule[0][env.kStep + 1]

            storage_soc = new_observation[3]
            new_observation = list(new_observation)
            new_observation[1] = cooling_load
            new_observation[8] = pv_power
            new_observation[9] = auxiliary_load
            new_observation = tuple(new_observation)

            new_observation = order_state_variables(env=env,
                                                    observation=new_observation,
                                                    cooling_load_predictions=cooling_load_predictions,
                                                    electricity_price_predictions=electricity_price_predictions,
                                                    pv_power_generation_predictions=pv_power_generation_predictions,
                                                    horizon=24,
                                                    step=episode_step)

            # Scale observations
            new_observation = min_max_scaling(new_observation, env.state_mins, env.state_maxs, np.array([0]),
                                              np.array([1]))

            # print(new_observation)
            agent.remember(observation, action, reward, new_observation, done)

            score += reward
            if episode != num_episodes:
                agent.learn()

            observation = new_observation

        score_history.append(score)

        print(f'Episode: {episode}, Score: {score}')