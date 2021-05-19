"""

"""

from eppy.modeleditor import IDF
import os
from GymEnvironments.environment_discrete_action import RelicEnv
import pandas as pd
from agents.RBC_discrete import RBCAgent
from utils import calculate_tank_soc, set_occupancy_schedule

# Run a simple RBC simulation in order to generate the new forcing variables resulting from the modification of the
# .idf file performed in the lines above. Even if the RBC does not obtain satisfying performance the only purpose of
# this step is to generate the forcing variables of the control problem which are not influenced by the control action.
directory = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':

    occupancy_schedule_index = 1

    result_directory = 'data/'
    min_temperature_limit = 10  # Below this value no charging
    min_charging_temperature = 12  # Charging begins above this threshold
    max_temperature_limit = 18  # Above this threshold no discharging

    price_schedule_name = 'electricity_price_schedule_base.csv'

    config = {
        'res_directory': result_directory,
        # Change this folder to the path where you want to save the output of each episode
        'weather_file': 'ITA_TORINO-CASELLE_IGDG',
        'simulation_days': 92,
        'tank_min_temperature': min_temperature_limit,
        'tank_max_temperature': max_temperature_limit,
        'end_day_of_month': 31,
        'price_schedule_name': price_schedule_name,
        'pv_nominal_power': 2000,
        'occupancy_schedule_index': occupancy_schedule_index}

    env = RelicEnv(config)

    # Import predictions
    # cooling_load_predictions = pd.read_csv('supportFiles\\prediction-cooling_load_perfect.csv')
    # electricity_price_predictions = pd.read_csv('supportFiles\\prediction-electricity_price_base_perfect.csv')
    electricity_price_schedule = pd.read_csv('supportFiles/electricity_price_schedule_base.csv', header=None)

    min_price = float(electricity_price_schedule[0].min())
    charge_flag = 0

    # evaluate SOC
    max_storage_soc = calculate_tank_soc(min_temperature_limit, min_temperature_limit,
                                         max_temperature_limit)  # The storage is full
    min_charging_storage_soc = calculate_tank_soc(min_charging_temperature, min_temperature_limit,
                                                  max_temperature_limit)
    min_storage_soc = calculate_tank_soc(max_temperature_limit, min_temperature_limit,
                                         max_temperature_limit)  # The storage is empty

    rbc_controller = RBCAgent(min_storage_soc=min_storage_soc,
                              min_charging_storage_soc=min_charging_storage_soc,
                              max_storage_soc=max_storage_soc,
                              min_electricity_price=min_price)

    # Define the number of episodes
    num_episodes = 1
    score_history = []

    # Training Loop
    for episode in range(1, num_episodes + 1):
        episode_step = 0
        observation = env.reset(name_save='episode')
        # append prediction
        electricity_price = electricity_price_schedule[0][env.kStep]
        storage_soc = observation[3]

        score = 0
        done = False

        while not done:

            action = rbc_controller.choose_action(electricity_price=electricity_price,
                                                  storage_soc=storage_soc)

            step = 1
            reward = 0
            while step <= env.ep_time_step:
                new_observation, reward_step, done, info = env.step(action)
                reward += reward_step
                step += 1
                # if done:
                #     break
            episode_step += 1

            if done:
                break

            # append predictions
            electricity_price = electricity_price_schedule[0][env.kStep]
            storage_soc = new_observation[3]

            # print(new_observation)

            observation = new_observation

        score_history.append(score)

        print(f'Episode: {episode}, Score: {score}')

    sim_data = pd.read_csv(result_directory + '\\episode_1.csv', sep=';', decimal=',') # Read .csv output file

    time = sim_data['EMS:currentTimeOfDay [](TimeStep)']
    cooling_load = sim_data['IDEALLOADPROFILE:Plant Load Profile Heat Transfer Rate [W](TimeStep)']
    ambient_temperature = sim_data['THERMAL ZONE 4:Zone Mean Air Temperature [C](TimeStep)']
    electricity_price = sim_data['Price']
    pv_production = sim_data['PV power generation [W]']

    series = [time, cooling_load, ambient_temperature, electricity_price, pv_production]

    forcing_var_data = pd.concat(series, axis=1)
    forcing_var_data.columns = ['time', 'cooling_load', 'ambient_temperature', 'electricity_price', 'pv_power_generation']
    forcing_var_data['cooling_load'] = forcing_var_data.cooling_load.abs()
    forcing_var_data.to_csv(path_or_buf='data\\forcing_variables_occ{}.csv'.format(str(occupancy_schedule_index)),
                            index=False)
