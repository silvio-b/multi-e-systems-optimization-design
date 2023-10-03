import os
from GymEnvironments.environment_discrete_action import RelicEnv
import pandas as pd
from agents.SAC_discrete import SACAgent
from utils import order_state_variables, min_max_scaling, calculate_tank_soc
import numpy as np
import json

directory = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':

    test_id = 'test_predictions_1'
    test_schedule = pd.read_csv('testSchedules'+'\\' + test_id + '.csv', decimal=',', sep=';')
    result_directory_path = 'D:\\Projects\\PhD_Silvio\\MultiEnergyOptimizationDesign\\DiscreteTests'

    for test in range(8, test_schedule.shape[0]):
        best_score = 1000
        result_directory = '\\' + test_id + '\\configuration' + test_schedule['id'][test]
        safe_exploration = -1
        discount_factor = 0.99
        alpha = test_schedule['alpha'][test]
        tau = 0.005
        automatic_entropy_tuning = False
        learning_rate_actor = test_schedule['lr'][test]
        learning_rate_critic = test_schedule['lr'][test]
        n_hidden_layers = test_schedule['dim'][test]
        n_neurons = test_schedule['neurons'][test]
        batch_size = test_schedule['batch_size'][test]
        replay_buffer_capacity = 24 * 30 * 100
        prediction_observations_index = test_schedule['predictions_observations_index'][test]

        prediction_horizon = test_schedule['prediction_horizon'][test]
        seed = test_schedule['seed'][test]
        occupancy_schedule_index = 0
        appliances = test_schedule['appliances'][test]
        price_schedule_type = test_schedule['price_schedule_type'][test]

        if prediction_observations_index == 0:
            prediction_observations = ['electricity_price', 'cooling_load', 'pv_power_generation']
        else:
            prediction_observations = ['electricity_price']

        # physical parameters
        min_temperature_limit = 10  # Below this value no charging
        min_charging_temperature = 12  # Charging begins above this threshold
        max_temperature_limit = 18  # Above this threshold no discharging

        pv_nominal_power = test_schedule['pv_nominal_power'][test]
        battery_size = test_schedule['battery_size'][test]

        tank_volume = test_schedule['tank_volume'][test]
        tank_heat_gain_coefficient = test_schedule['tank_heat_gain_coefficient'][test]

        # price schedule
        price_schedule_name = 'electricity_price_schedule_{}.csv'.format(price_schedule_type)
        electricity_price_schedule = pd.read_csv('supportFiles\\' + price_schedule_name, header=None)
        min_price = float(electricity_price_schedule[0].min())
        max_price = float(electricity_price_schedule[0].max())

        num_episodes = test_schedule['num_episodes'][test]
        # num_episodes = 2
        result_directory_final = result_directory_path + result_directory
        if not os.path.exists(result_directory_final):
            os.makedirs(result_directory_final)

        with open('supportFiles\\state_space_variables.json', 'r') as json_file:
            building_states = json.load(json_file)

        building_states['predicted_observations']['horizon'] = int(prediction_horizon)
        building_states['predicted_observations']['variables'] = prediction_observations

        if appliances == 0:
            building_states['eplus_observations']["auxiliary_load"] = "False"
        else:
            building_states['eplus_observations']["auxiliary_load"] = "True"

        with open('supportFiles\\state_space_variables.json', 'w') as json_file:
            json.dump(building_states, json_file)

        with open('supportFiles\\state_rescaling_table.json', 'r') as json_file_2:
            state_rescaling = json.load(json_file_2)

        state_rescaling['pv_power_generation']['max'] = float(pv_nominal_power)
        state_rescaling['electricity_price']['min'] = min_price
        state_rescaling['electricity_price']['max'] = max_price

        with open('supportFiles\\state_rescaling_table.json', 'w') as json_file_2:
            json.dump(state_rescaling, json_file_2)

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
            'pv_nominal_power': pv_nominal_power,
            'battery_size': battery_size,
            'price_schedule_name': price_schedule_name,
            'occupancy_schedule_index': occupancy_schedule_index,
            'appliances': appliances}

        env = RelicEnv(config)

        # Import predictions
        cooling_load_predictions = pd.read_csv('supportFiles\\prediction-cooling_load_perfect_occ{}.csv'.format(
            occupancy_schedule_index
        ))
        electricity_price_predictions = pd.read_csv('supportFiles/prediction-electricity_price_{}_perfect.csv'.format(
            price_schedule_type
        ))
        pv_power_generation_predictions = pd.read_csv('supportFiles\\prediction-pv_power_generation_perfect_{}.csv'.format(
            str(pv_nominal_power)
        ))

        # Set the number of actions
        n_actions = 3
        input_dims = env.observation_space.shape[0]

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
                         reward_scaling=10., seed=seed, rbc_controller=None, safe_exploration=safe_exploration,
                         automatic_entropy_tuning=automatic_entropy_tuning, alpha=alpha)

        # Define the number of episodes
        score_history = []
        done = False
        # baseline simulation
        #

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
                                                horizon=prediction_horizon,
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
                                                        horizon=prediction_horizon,
                                                        step=episode_step)

                # Scale observations
                new_observation = min_max_scaling(new_observation, env.state_mins, env.state_maxs, np.array([0]),
                                                  np.array([1]))
                # true_action = info['true_action'][0]
                # print(new_observation)
                agent.remember(observation, action, reward, new_observation, done)

                score += reward
                if episode != num_episodes:
                    agent.learn()

                observation = new_observation

            score_history.append(score)

            if env.episode_electricity_cost < best_score:
                best_score = env.episode_electricity_cost
                best_episode = episode
                agent.save_models(path=result_directory_final)

            print(f'Episode: {episode}, Score: {score}')

        last_episode_cost = env.episode_electricity_cost

        # Deploy for 1 episode with different occupancy
        agent.load_models(path=result_directory_final)

        occupancy_schedule_index = 1

        config = {
            'res_directory': result_directory_final,
            # Change this folder to the path where you want to save the output of each episode
            'weather_file': 'ITA_TORINO-CASELLE_IGDG',
            'simulation_days': 90,
            'tank_min_temperature': min_temperature_limit,
            'tank_max_temperature': max_temperature_limit,
            'tank_volume': tank_volume,
            'tank_heat_gain_coefficient': tank_heat_gain_coefficient,
            'pv_nominal_power': pv_nominal_power,
            'battery_size': battery_size,
            'price_schedule_name': price_schedule_name,
            'occupancy_schedule_index': occupancy_schedule_index,
            'appliances': appliances}

        env = RelicEnv(config)

        cooling_load_predictions = pd.read_csv('supportFiles\\prediction-cooling_load_perfect_occ{}.csv'.format(
            occupancy_schedule_index
        ))

        episode_step = 0
        observation = env.reset(name_save='deploy')
        # append prediction
        electricity_price = electricity_price_schedule[0][env.kStep + 1]
        storage_soc = observation[3]
        observation = order_state_variables(env_names=env.state_names,
                                            observation=observation,
                                            cooling_load_predictions=cooling_load_predictions,
                                            electricity_price_predictions=electricity_price_predictions,
                                            pv_power_generation_predictions=pv_power_generation_predictions,
                                            horizon=prediction_horizon,
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
                                                    horizon=prediction_horizon,
                                                    step=episode_step)

            # Scale observations
            new_observation = min_max_scaling(new_observation, env.state_mins, env.state_maxs, np.array([0]),
                                              np.array([1]))
            # true_action = info['true_action'][0]
            # print(new_observation)
            # agent.remember(observation, action, reward, new_observation, done)

            score += reward

            observation = new_observation

        deploy_cost = env.episode_electricity_cost

        test_schedule['score'][test] = last_episode_cost
        test_schedule['best_score'][test] = best_score
        test_schedule['best_episode'][test] = best_episode
        test_schedule['deploy'][test] = deploy_cost

        test_schedule.to_csv(result_directory_path + '\\' + test_id + '.csv', decimal=',', sep=';', index=False)
