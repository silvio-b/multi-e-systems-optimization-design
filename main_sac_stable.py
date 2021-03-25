import os
from GymEnvironments.environment_continuous_stable import RelicEnv
import json

from stable_baselines3 import SAC

result_directory_path = 'D:\\OneDrive - Politecnico di Torino\\PhD_Silvio\\14_Projects\\002_PVZenControl\\Thermal_Electrical_Storage_Control\\'
result_directory = 'test_07'
safe_exploration = -1
discount_factor = 0.99
alpha = 0.1
tau = 0.005
automatic_entropy_tuning = True
learning_rate_actor = 0.005
learning_rate_critic = 0.005
n_hidden_layers = 2
n_neurons = 256
batch_size = 256
replay_buffer_capacity = 24 * 30 * 100
prediction_observations = ['electricity_price', 'pv_power_generation', 'cooling_load']
prediction_horizon = 0
min_temperature_limit = 10  # Below this value no charging
min_charging_temperature = 12  # Charging begins above this threshold
max_temperature_limit = 18  # Above this threshold no discharging
num_episodes = 30

result_directory_final = result_directory_path + result_directory
if not os.path.exists(result_directory_final):
    os.makedirs(result_directory_final)

config = {
    'res_directory': result_directory_final,
    # Change this folder to the path where you want to save the output of each episode
    'weather_file': 'ITA_TORINO-CASELLE_IGDG',
    'simulation_days': 90,
    'tank_min_temperature': min_temperature_limit,
    'tank_max_temperature': max_temperature_limit,
    'price_schedule_name': 'electricity_price_schedule.csv',
    'horizon': prediction_horizon}

with open('supportFiles//state_space_variables.json', 'r') as json_file:
    building_states = json.load(json_file)

building_states['predicted_observations']['horizon'] = int(prediction_horizon)
building_states['predicted_observations']['variables'] = prediction_observations

with open('supportFiles//state_space_variables.json', 'w') as json_file:
    json.dump(building_states, json_file)

env = RelicEnv(config)

model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0001)
model.learn(total_timesteps=90*24*10, log_interval=4)