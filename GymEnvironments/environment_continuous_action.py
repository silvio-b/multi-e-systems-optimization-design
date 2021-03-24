import os
import time as tm
import math
import gym
import numpy as np
import pandas as pd
from pyEp import pyEp
from gym import spaces
import json
from utils import get_state_variables, get_eplus_action_encoding, calculate_tank_soc
from eppy.modeleditor import IDF
from EnergyModels.PVmodel import PV
from EnergyModels.battery_model import Battery


class RelicEnv(gym.Env):

    def __init__(self, config):

        self.directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        self.idf_name = 'eplusModel'

        # directory where csv of each episode are saved
        if 'res_directory' in config:
            self.res_directory = config['res_directory']
        else:
            self.res_directory = ''

        print("File directory: ", self.directory)
        # EnergyPlus weather file

        if "idf_directory_name" in config:
            self.idf_directory_name = config["idf_directory_name"]
        else:
            self.idf_directory_name = 'StorageTank_Model'

        if "weather_file" in config:
            self.weather_file = config["weather_file"]
        else:
            self.weather_file = 'weatherFiles/ITA_TORINO-CASELLE_IGDG'

        # EnergyPlus TimeStep as specified in .idf file
        if "ep_time_step" in config:
            self.ep_time_step = config["ep_time_step"]
        else:
            self.ep_time_step = 12

        # Number of days of a simulation as specified in the RunPeriod object of the .idf file
        if "simulation_days" in config:
            self.simulation_days = config["simulation_days"]
        else:
            self.simulation_days = 90

        if "reward_multiplier" in config:
            self.reward_multiplier = config["reward_multiplier"]
        else:
            self.reward_multiplier = 10

        # Tank properties
        if "tank_volume" in config:
            self.tank_volume = config["tank_volume"]
        else:
            self.tank_volume = 10

        if "tank_heat_gain_coefficient" in config:
            self.tank_heat_gain_coefficient = config["tank_heat_gain_coefficient"]
        else:
            self.tank_heat_gain_coefficient = 12

        # Minimum Temperature allowed for water in the thermal storage tank.
        if "tank_min_temperature" in config:
            self.tank_min_temperature = config["tank_min_temperature"]
        else:
            self.tank_min_temperature = 10

        # Maximum Temperature allowed for water in the thermal storage tank.
        if "tank_max_temperature" in config:
            self.tank_max_temperature = config["tank_max_temperature"]
        else:
            self.tank_max_temperature = 18

        if "begin_month" in config:
            self.begin_month = config["begin_month"]
        else:
            self.begin_month = 6

        if "end_month" in config:
            self.end_month = config["end_month"]
        else:
            self.end_month = 8

        if "begin_day_of_month" in config:
            self.begin_day_of_month = config["begin_day_of_month"]
        else:
            self.begin_day_of_month = 1

        if "end_day_of_month" in config:
            self.end_day_of_month = config["end_day_of_month"]
        else:
            self.end_day_of_month = 29

        if "day_shift" in config:
            self.day_shift = config["day_shift"]
        else:
            self.day_shift = 152

        if "pv_surface" in config:
            self.pv_surface = config["pv_surface"]
        else:
            self.pv_surface = 10

        if "battery_size" in config:
            self.battery_size = config["battery_size"]
        else:
            self.battery_size = 2400

        idd_file = 'supportFiles\\Energy+9-2-0.idd'
        file_name = 'EplusModels\\' + self.idf_directory_name + '\\eplusModel.idf'

        IDF.setiddname(idd_file)
        idf_file = IDF(file_name)

        runperiod = idf_file.idfobjects['RUNPERIOD'][0]
        runperiod.Begin_Month = self.begin_month
        runperiod.Begin_Day_of_Month = self.begin_day_of_month
        runperiod.End_Month = self.end_month
        runperiod.End_Day_of_Month = self.end_day_of_month

        storage_tank = idf_file.idfobjects['ThermalStorage:ChilledWater:Mixed'][0]
        storage_tank.Tank_Volume = self.tank_volume
        storage_tank.Heat_Gain_Coefficient_from_Ambient_Temperature = self.tank_heat_gain_coefficient

        idf_file.save('eplusModels\\' + self.idf_directory_name + '\\eplusModel.idf', encoding='UTF-8')

        # EnergyPlus path in the local machine
        self.eplus_path = 'C:/EnergyPlusV9-2-0/'

        # Number of steps per day
        self.DAYSTEPS = int(24 * self.ep_time_step)

        # Total number of steps
        self.MAXSTEPS = int(self.simulation_days * self.DAYSTEPS)

        # Time difference between each step in seconds
        self.deltaT = (60 / self.ep_time_step) * 60

        # Outputs given by EnergyPlus, defined in variables.cfg
        self.outputs = []

        # Inputs expected by EnergyPlus, defined in variables.cfg
        self.inputs = []

        # Current step of the simulation
        self.kStep = 0

        # Instance of EnergyPlus simulation
        self.ep = None

        # state can be all the inputs required to make a control decision
        # getting all the outputs coming from EnergyPlus for the time being
        with open(self.directory + '\\supportFiles\\state_space_variables.json') as json_file:
            buildings_states = json.load(json_file)

        with open(self.directory + '\\supportFiles\\state_rescaling_table.json') as json_file:
            rescaling_table = json.load(json_file)

        self.state_types, self.state_names, self.state_mins, self.state_maxs = \
            get_state_variables(data=buildings_states,
                                rescaling_table=rescaling_table)

        self.state_mins = np.array(self.state_mins)
        self.state_maxs = np.array(self.state_maxs)

        self.observation_space = spaces.Box(np.repeat(0, len(self.state_mins)),
                                            np.repeat(0, len(self.state_maxs)),
                                            dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.episode_number = 1

        # Price of the electricity schedule
        self.electricity_price_schedule = pd.read_csv(self.directory + '\\supportFiles\\electricity_price_schedule.csv',
                                                      header=None)
        self.max_price = self.electricity_price_schedule.values.max()

        # PV & Battery Initialization
        self.pv = PV(surface=self.pv_surface, tilt_angle=40, azimuth=180 - 64)

        self.battery = Battery(max_power=1500, max_capacity=self.battery_size, rte=0.96)

        self.eta_ac_dc = 0.9

        self.SOC = 1

        # Lists for adding variables to eplus output (.csv)
        self.action_list = []
        self.reward_list = []
        self.price_list = []
        self.tank_soc_list = []
        self.battery_soc_list = []
        self.incidence_list = []
        self.zenith_list = []
        self.efficiency_list = []
        self.pv_power_generation_list = []
        self.pv_energy_production_list = []
        self.pv_energy_to_building_list = []
        self.pv_energy_to_grid_list = []
        self.pv_energy_to_battery_list = []
        self.grid_energy_list = []
        self.grid_energy_to_building_list = []
        self.grid_energy_to_battery_list = []
        self.battery_energy_to_building_list = []
        self.p_cool_list = []
        self.building_energy_consumption_list = []
        self.savings_list = []
        self.grid_list = []
        self.energy_cost_from_grid_list = []
        self.energy_cost_to_grid_list = []


    def step(self, action: np.ndarray):

        # current time from start of simulation
        time = self.kStep * self.deltaT
        # current time from start of day
        dayTime = time % 86400
        day_of_the_year = (time // 86400) + self.day_shift
        if dayTime == 0:
            print("Day: ", int(self.kStep / self.DAYSTEPS))

        # prendo le azioni dal controllore
        action = action[0]

        if self.SOC == 0 and action < 0: # AVOID discharge when SOC is 0
            action = 0
            eplus_commands = get_eplus_action_encoding(action=action)
        else:
            eplus_commands = get_eplus_action_encoding(action=action)

        # EPlus simulation, input packet construction and feeding to Eplus
        self.inputs = eplus_commands
        input_packet = self.ep.encode_packet_simple(self.inputs, time)
        self.ep.write(input_packet)

        # after EnergyPlus runs the simulation step, it returns the outputs
        output_packet = self.ep.read()
        self.outputs = self.ep.decode_packet_simple(output_packet)

        # Append agent action, may differ from actual actions
        self.action_list.append(action)
        # print("Outputs:", self.outputs)
        if not self.outputs:
            print("Outputs:", self.outputs)
            print("Actions:", action)
            next_state = self.reset()
            return next_state, 0, False, {}

        # Unpack Eplus output

        # CALCULATE soc
        for i in range(5, 10):
            self.outputs[i] = calculate_tank_soc(self.outputs[i],
                                                 min_temperature=self.tank_min_temperature,
                                                 max_temperature=self.tank_max_temperature)

        time, day, outdoor_air_temperature, cooling_load, chiller_energy_consumption, storage_soc, storage_soc_l1, \
        storage_soc_l2, storage_soc_l3, storage_soc_l4, diff_i, dir_i, auxiliary_energy_consumption, \
        pump_energy_consumption = self.outputs

        self.SOC = storage_soc

        building_energy_consumption_ac = chiller_energy_consumption + pump_energy_consumption + \
                                         auxiliary_energy_consumption

        # PV model from PV class, PV power in W, PV energy in Joule
        incidence, zenith = self.pv.solar_angles_calculation(day=day_of_the_year, time=time)
        pv_power, efficiency = self.pv.electricity_prediction(direct_radiation=dir_i, diffuse_radiation=diff_i,
                                                              day=day_of_the_year, time=time,
                                                              t_out=outdoor_air_temperature)

        pv_energy_production_dc = pv_power * 60 * 60 / self.ep_time_step

        pv_energy_excess_dc = pv_energy_production_dc - building_energy_consumption_ac / self.eta_ac_dc
        # PV has always priority
        # agent_battery_energy = abs(action_batt * self.battery.max_power * 60 * 60 / self.epTimeStep)

        max_charge_dc = min(self.battery.max_power * 60 * 60 / self.ep_time_step,
                            (self.battery.soc_max - self.battery.soc) * self.battery.max_capacity * 3600 /
                            self.battery.eta_dc_dc)
        max_discharge_dc = min(self.battery.max_power * 60 * 60 / self.ep_time_step,
                               (self.battery.soc - self.battery.soc_min) * self.battery.max_capacity * 3600 *
                               self.battery.eta_dc_dc)

        electricity_price = self.electricity_price_schedule[0][self.kStep]

        if pv_energy_excess_dc > 0:

            battery_energy_to_building_ac = 0
            pv_energy_to_building_ac = building_energy_consumption_ac
            grid_energy_to_building_ac = 0
            grid_energy_to_battery_ac = 0

            if pv_energy_excess_dc > max_charge_dc:
                net_battery_energy_dc = max_charge_dc
                pv_energy_to_grid_ac = (pv_energy_excess_dc - max_charge_dc) * self.eta_ac_dc
            else:
                net_battery_energy_dc = pv_energy_excess_dc
                pv_energy_to_grid_ac = 0

            self.battery.charge(net_battery_energy_dc)
            pv_energy_to_battery_dc = net_battery_energy_dc

        else:

            pv_energy_to_building_ac = pv_energy_production_dc * self.eta_ac_dc
            pv_energy_to_battery_dc = 0
            pv_energy_to_grid_ac = 0
            grid_energy_to_battery_ac = 0

            if electricity_price >= self.max_price:
                building_energy_residual_ac = building_energy_consumption_ac - pv_energy_to_building_ac

                if building_energy_residual_ac / self.eta_ac_dc <= max_discharge_dc:
                    net_battery_energy_dc = building_energy_consumption_ac / self.eta_ac_dc - pv_energy_production_dc

                    battery_energy_to_building_ac = net_battery_energy_dc * self.eta_ac_dc
                    grid_energy_to_building_ac = 0
                else:
                    net_battery_energy_dc = max_discharge_dc
                    battery_energy_to_building_ac = net_battery_energy_dc * self.eta_ac_dc
                    grid_energy_to_building_ac = building_energy_consumption_ac - pv_energy_to_building_ac - \
                                                 battery_energy_to_building_ac
            else:
                net_battery_energy_dc = 0
                battery_energy_to_building_ac = 0
                pv_energy_to_grid_ac = 0
                grid_energy_to_building_ac = building_energy_consumption_ac - pv_energy_to_building_ac

            self.battery.discharge(net_battery_energy_dc)

        grid_energy_ac = grid_energy_to_building_ac + grid_energy_to_battery_ac

        self.battery.soc = np.clip(self.battery.soc, self.battery.soc_min, self.battery.soc_max)

        # START REWARD CALCULATIONS
        energy_cost_from_grid = - (grid_energy_ac / (3.6 * 1000000) * electricity_price)
        energy_cost_to_grid = (pv_energy_to_grid_ac / (3.6 * 1000000) * 0.001)

        reward_price = energy_cost_from_grid + energy_cost_to_grid

        # price component
        reward = reward_price
        reward = reward * self.reward_multiplier
        # END REWARD CALCULATIONS

        self.reward_list.append(reward)
        self.price_list.append(electricity_price)
        self.tank_soc_list.append(storage_soc)
        self.battery_soc_list.append(self.battery.soc)
        self.incidence_list.append(incidence)
        self.zenith_list.append(zenith)
        self.efficiency_list.append(efficiency)
        self.pv_power_generation_list.append(pv_power)
        self.pv_energy_production_list.append(pv_energy_production_dc)
        self.pv_energy_to_building_list.append(pv_energy_to_building_ac)
        self.pv_energy_to_battery_list.append(pv_energy_to_battery_dc)
        self.pv_energy_to_grid_list.append(pv_energy_to_grid_ac)
        self.grid_energy_list.append(grid_energy_ac)
        self.grid_energy_to_building_list.append(grid_energy_to_building_ac)
        self.grid_energy_to_battery_list.append(grid_energy_to_battery_ac)
        self.battery_energy_to_building_list.append(battery_energy_to_building_ac)
        self.building_energy_consumption_list.append(building_energy_consumption_ac)
        self.p_cool_list.append(cooling_load)
        self.energy_cost_from_grid_list.append(energy_cost_from_grid)
        self.energy_cost_to_grid_list.append(energy_cost_to_grid)

        # the cooling load is returned as negative value from energy plus
        cooling_load = np.abs(cooling_load)

        next_state = (outdoor_air_temperature, cooling_load, electricity_price, storage_soc, storage_soc_l1,
                      storage_soc_l2, storage_soc_l3, storage_soc_l4, pv_power, auxiliary_energy_consumption,
                      self.battery.soc)
        self.kStep += 1

        done = False
        if self.kStep >= self.MAXSTEPS:
            print(self.kStep)
            # requires one more step to close the simulation
            input_packet = self.ep.encode_packet_simple(self.inputs, time)
            self.ep.write(input_packet)
            # output is empty in the final step
            # but it is required to read this output for termination
            output_packet = self.ep.read()
            last_output = self.ep.decode_packet_simple(output_packet)
            print("Finished simulation")
            print("Last action: ", action)
            print("Last reward: ", reward)
            done = True
            print(done)

            self.ep.close()
            tm.sleep(20)

            dataep = pd.read_csv(self.directory + '\\eplusModels\\storageTank_Model\\' + self.idf_name + '.csv')
            dataep['Action'] = self.action_list
            dataep['Reward'] = self.reward_list
            dataep['Price'] = self.price_list
            dataep['Tank SOC'] = self.tank_soc_list
            dataep['Battery soc'] = self.battery_soc_list
            dataep['Incidence'] = self.incidence_list
            dataep['Zenith'] = self.zenith_list
            dataep['Efficiency'] = self.efficiency_list
            dataep['PV power generation [W]'] = self.pv_power_generation_list
            dataep['PV energy production [J]'] = self.pv_energy_production_list
            dataep['PV energy to building [J]'] = self.pv_energy_to_building_list
            dataep['PV energy to grid [J]'] = self.pv_energy_to_grid_list
            dataep['PV energy to battery [J]'] = self.pv_energy_to_battery_list
            dataep['grid energy [J]'] = self.grid_energy_list
            dataep['grid energy to building [J]'] = self.grid_energy_to_building_list
            dataep['grid energy to battery [J]'] = self.grid_energy_to_battery_list
            dataep['battery energy to building [J]'] = self.battery_energy_to_building_list
            dataep['Cooling load [W]'] = self.p_cool_list
            dataep['Building load [J]'] = self.building_energy_consumption_list
            dataep['Energy costs from grid [€]'] = self.energy_cost_from_grid_list
            dataep['Energy costs to grid [€]'] = self.energy_cost_to_grid_list

            episode_electricity_consumption = dataep['CHILLER:Chiller Electric Energy [J](TimeStep)'].sum() / (
                        3.6 * 1000000)
            episode_electricity_cost = dataep['Energy costs from grid [€]'].sum() - dataep['Energy costs to grid [€]'].sum()
            print('Elec consumption: ' + str(episode_electricity_consumption) +
                  ' Elec Price: ' + str(episode_electricity_cost))

            dataep.to_csv(path_or_buf=self.res_directory + '/' + 'episode_' + str(self.episode_number) + '.csv',
                          sep=';', decimal=',', index=False)
            self.episode_number = self.episode_number + 1
            self.ep = None
            self.action_list = []
            self.reward_list = []
            self.price_list = []
            self.tank_soc_list = []
            self.battery_soc_list = []
            self.incidence_list = []
            self.zenith_list = []
            self.efficiency_list = []
            self.pv_power_generation_list = []
            self.pv_energy_production_list = []
            self.pv_energy_to_building_list = []
            self.pv_energy_to_grid_list = []
            self.pv_energy_to_battery_list = []
            self.grid_energy_list = []
            self.grid_energy_to_building_list = []
            self.grid_energy_to_battery_list = []
            self.battery_energy_to_building_list = []
            self.p_cool_list = []
            self.building_energy_consumption_list = []
            self.grid_list = []
            self.energy_cost_from_grid_list = []
            self.energy_cost_to_grid_list = []
            self.SOC = 1
            self.battery.soc = 0.46

        info = {}
        if self.kStep == self.MAXSTEPS:
            print(done)
        return next_state, reward, done, info

    def reset(self):
        # stop existing energyplus simulation
        if self.ep:
            print("Closing the old simulation and socket.")
            self.ep.close()  # needs testing: check if it stops the simulation
            tm.sleep(5)
            self.ep = None

        # tm.sleep(30)
        # start new simulation
        print("Starting a new simulation..")
        self.kStep = 0
        pyEp.set_eplus_dir("C:\\EnergyPlusV9-2-0")
        path_to_buildings = os.path.join(self.directory, 'eplusModels')
        builder = pyEp.socket_builder(path_to_buildings)
        configs = builder.build()
        self.ep = pyEp.ep_process('localhost', configs[0][0], configs[0][1], self.weather_file)

        self.outputs = np.round(self.ep.decode_packet_simple(self.ep.read()), 1).tolist()

        for i in range(5, 10):
            self.outputs[i] = calculate_tank_soc(self.outputs[i],
                                                 min_temperature=self.tank_min_temperature,
                                                 max_temperature=self.tank_max_temperature)

        time, day, outdoor_air_temperature, cooling_load, chiller_energy_consumption, storage_soc, storage_soc_l1, \
        storage_soc_l2, storage_soc_l3, storage_soc_l4, diff_i, dir_i, auxiliary_energy_consumption, \
        pump_energy_consumption = self.outputs

        pv_power, efficiency = self.pv.electricity_prediction(direct_radiation=dir_i, diffuse_radiation=diff_i,
                                                              day=self.day_shift, time=time,
                                                              t_out=outdoor_air_temperature)

        self.SOC = storage_soc

        electricity_price = self.electricity_price_schedule[0][self.kStep]

        # the cooling load is returned as negative value from energy plus
        cooling_load = np.abs(cooling_load)

        next_state = (outdoor_air_temperature, cooling_load, electricity_price, storage_soc, storage_soc_l1,
                      storage_soc_l2, storage_soc_l3, storage_soc_l4, pv_power, auxiliary_energy_consumption,
                      self.battery.soc)

        return next_state

    def render(self, mode='human', close=False):
        pass
