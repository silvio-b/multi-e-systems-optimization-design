import os
from GymEnvironments.environment_discrete_action import RelicEnv
import pandas as pd
from agents.RBC_discrete import RBCAgent
from utils import calculate_tank_soc

directory = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':

    result_directory_path = 'D:\\OneDrive - Politecnico di Torino\\PhD_Silvio\\14_Projects\\002_PVZenControl\\Thermal_Electrical_Storage_Control\\'
    result_directory = 'baseline'
    min_temperature_limit = 10  # Below this value no charging
    min_charging_temperature = 12  # Charging begins above this threshold
    max_temperature_limit = 18  # Above this threshold no discharging
    occupancy_schedule_index = 0

    result_directory_final = result_directory_path + result_directory
    if not os.path.exists(result_directory_final):
        os.makedirs(result_directory_final)

    config = {
        'res_directory': result_directory_final,
        # Change this folder to the path where you want to save the output of each episode
        'weather_file': 'ITA_TORINO-CASELLE_IGDG',
        'simulation_days': 90,
        'tank_min_temperature': min_temperature_limit,
        'tank_max_temperature': max_temperature_limit}

    env = RelicEnv(config)

    # configure occupancy schedule
    # idf_dir = 'eplusModels\\StorageTank_Model'
    # idd_file = 'supportFiles\\Energy+9-2-0.idd'
    # file_name = idf_dir + '\\eplusModel.idf'
    # IDF.setiddname(idd_file)
    # idf_file = IDF(file_name)
    # occupancy_schedule = idf_file.idfobjects['Schedule:Compact'][0]
    # occupancy_schedule = set_occupancy_schedule(occupancy_schedule, occupancy_schedule_index)
    # idf_file.save(idf_dir + '\\eplusModel.idf', encoding='UTF-8')
    # end configuration occupancy schedule

    # Import predictions
    cooling_load_predictions = pd.read_csv('supportFiles\\prediction-cooling_load_perfect.csv')
    electricity_price_predictions = pd.read_csv('supportFiles/prediction-electricity_price_base_perfect.csv')
    electricity_price_schedule = pd.read_csv('supportFiles\\electricity_price_schedule.csv', header=None)

    min_price = float(electricity_price_schedule[0].min())
    charge_flag = 0
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
        observation = env.reset()
        # append prediction
        electricity_price = electricity_price_schedule[0][env.kStep + 1]
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
            electricity_price = electricity_price_schedule[0][env.kStep + 1]
            storage_soc = new_observation[3]

            # print(new_observation)

            observation = new_observation

        score_history.append(score)

        print(f'Episode: {episode}, Score: {score}')
