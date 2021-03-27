import os
from GymEnvironments.environment_discrete_action_multi import RelicEnv
from GymEnvironments.environment_discrete_action import RelicEnv as RelicEnvBaseline
import pandas as pd
from agents.SAC_discrete import SACAgent
from agents.RBC_discrete import RBCAgent
from utils import order_state_variables, min_max_scaling, calculate_tank_soc
import numpy as np
import json

directory = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':

    test_id = 'test_01'
    test_schedule = pd.read_csv('testSchedules'+'\\'+ test_id + '.csv', decimal=',', sep=';')
    result_directory_path = 'D:\\Projects\\PhD_Silvio\\MultiEnergyOptimizationDesign\\SAC_Offline'

    for test in range(0, test_schedule.shape[0]):
        best_score = 1000
        result_directory = '\\' + test_id + '\\configuration' + test_schedule['id'][test]
        safe_exploration = -1
        discount_factor = 0.99
        alpha = test_schedule['alpha'][test]
        tau = 0.005
        automatic_entropy_tuning = False
        learning_rate_actor = test_schedule['lr'][test]
        learning_rate_critic = test_schedule['lr'][test]
        n_hidden_layers = 2
        n_neurons = 256
        batch_size = 64
        replay_buffer_capacity = 24 * 30 * 100
        prediction_observations = ['electricity_price', 'cooling_load', 'pv_power_generation']
        prediction_horizon = 24

        # physical parameters
        min_temperature_limit = 10  # Below this value no charging
        min_charging_temperature = 12  # Charging begins above this threshold
        max_temperature_limit = 18  # Above this threshold no discharging

        pv_surface = test_schedule['pv_surface'][test]
        battery_size = test_schedule['battery_size'][test]

        tank_volume = test_schedule['tank_volume'][test]
        tank_heat_gain_coefficient = test_schedule['tank_heat_gain_coefficient'][test]

        # price schedule
        price_schedule_name = 'electricity_price_schedule.csv'

        num_episodes = 20

        result_directory_final = result_directory_path + result_directory
        if not os.path.exists(result_directory_final):
            os.makedirs(result_directory_final)

        with open('supportFiles\\state_space_variables.json', 'r') as json_file:
            building_states = json.load(json_file)

        building_states['predicted_observations']['horizon'] = int(prediction_horizon)
        building_states['predicted_observations']['variables'] = prediction_observations

        with open('supportFiles\\state_space_variables.json', 'w') as json_file:
            json.dump(building_states, json_file)

        hidden_size = n_hidden_layers * [n_neurons]

        config = {
            'res_directory': result_directory_final,
            # Change this folder to the path where you want to save the output of each episode
            'weather_file': 'ITA_TORINO-CASELLE_IGDG',
            'simulation_days': 90,
            'tank_min_temperature': min_temperature_limit,
            'tank_max_temperature': max_temperature_limit,
            'tank_volume': tank_volume,
            'tank_heat_gain_coefficient': tank_heat_gain_coefficient,
            'pv_surface': pv_surface,
            'battery_size': battery_size,
            'price_schedule_name': price_schedule_name}

        env = RelicEnv(config)
        env_baseline = RelicEnvBaseline(config)

        # Import predictions
        cooling_load_predictions = pd.read_csv('supportFiles\\prediction-cooling_load_perfect.csv')
        electricity_price_predictions = pd.read_csv('supportFiles\\prediction-electricity_price_perfect.csv')
        pv_power_generation_predictions = pd.read_csv('supportFiles\\prediction-pv_power_generation_perfect.csv')
        electricity_price_schedule = pd.read_csv('supportFiles\\' + price_schedule_name, header=None)

        # Set the number of actions
        n_actions = 4
        input_dims = env.observation_space.shape[0]

        # define period for RBC control and
        min_price = float(electricity_price_schedule[0].min())

        # evaluate SOC
        max_storage_soc = calculate_tank_soc(min_temperature_limit, min_temperature_limit,
                                             max_temperature_limit)  # The storage is full
        min_charging_storage_soc = calculate_tank_soc(min_charging_temperature, min_temperature_limit,
                                                      max_temperature_limit)
        min_storage_soc = calculate_tank_soc(max_temperature_limit, min_temperature_limit,
                                             max_temperature_limit)  # The storage is empty

        # Initialize agent
        agent = SACAgent(state_dim=input_dims,
                         action_dim=n_actions, hidden_dim=hidden_size, discount=discount_factor, tau=tau,
                         lr_critic=learning_rate_critic, lr_actor=learning_rate_actor,
                         batch_size=batch_size, replay_buffer_capacity=replay_buffer_capacity, learning_start=30 * 24,
                         reward_scaling=10., seed=0, rbc_controller=None, safe_exploration=safe_exploration,
                         automatic_entropy_tuning=automatic_entropy_tuning, alpha=alpha)

        rbc_controller = RBCAgent(min_storage_soc=min_storage_soc,
                                  min_charging_storage_soc=min_charging_storage_soc,
                                  max_storage_soc=max_storage_soc,
                                  min_electricity_price=min_price)

        # Define the number of episodes
        score_history = []
        done = False
        # baseline simulation
        #
        observation = env_baseline.reset(name_save='baseline')
        # append prediction
        electricity_price = electricity_price_schedule[0][env_baseline.kStep + 1]
        storage_soc = observation[3]

        while not done:

            action = rbc_controller.choose_action(electricity_price=electricity_price,
                                                  storage_soc=storage_soc)

            step = 1
            reward = 0
            while step <= env_baseline.ep_time_step:
                new_observation, reward_step, done, info = env_baseline.step(action)
                reward += reward_step
                step += 1
                # if done:
                #     break

            if done:
                break

            # append predictions
            electricity_price = electricity_price_schedule[0][env_baseline.kStep + 1]
            storage_soc = new_observation[3]

            # print(new_observation)

            observation = new_observation

        baseline_cost = env_baseline.episode_electricity_cost

        # Training Loop
        for episode in range(1, num_episodes + 1):
            episode_step = 0
            observation = env.reset(name_save='episode')
            # append prediction
            electricity_price = electricity_price_schedule[0][env.kStep + 1]
            storage_soc = observation[3]
            observation = order_state_variables(env_names=env.state_names,
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

                new_observation = order_state_variables(env_names=env.state_names,
                                                        observation=new_observation,
                                                        cooling_load_predictions=cooling_load_predictions,
                                                        electricity_price_predictions=electricity_price_predictions,
                                                        pv_power_generation_predictions=pv_power_generation_predictions,
                                                        horizon=24,
                                                        step=episode_step)

                # Scale observations
                new_observation = min_max_scaling(new_observation, env.state_mins, env.state_maxs, np.array([0]),
                                                  np.array([1]))
                true_action = info['true_action'][0]
                # print(new_observation)
                agent.remember(observation, true_action, reward, new_observation, done)

                score += reward
                if episode != num_episodes:
                    agent.learn()

                observation = new_observation

            score_history.append(score)

            if env.episode_electricity_cost < best_score:
                best_score = env.episode_electricity_cost

            print(f'Episode: {episode}, Score: {score}')

        last_episode_cost = env.episode_electricity_cost

        test_schedule['score'][test] = last_episode_cost
        test_schedule['best_score'][test] = best_score
        test_schedule['baseline'][test] = baseline_cost

        agent.save_models(path=result_directory_final)

        test_schedule.to_csv(result_directory_path + '\\' + test_id + '.csv', decimal=',', sep=';', index=False)
