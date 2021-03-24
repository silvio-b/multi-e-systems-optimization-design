import re
import numpy as np
import pandas as pd
import math


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_state_variables(data: dict, rescaling_table: dict):
    """
    Method to return states type and names from configuration dictionary
    :param rescaling_table: dict
    :param data: dict
    :return:
    """
    states_type = []
    states_name = []
    for uid in data.keys():
        for var in data[uid].keys():
            if (uid == "eplus_observations") and (data[uid][var] == "True"):
                states_type.append(var)
                states_name.append(var + '_0')

            elif uid == "lagged_observations" and data[uid][var] > 0:
                for i in range(0, data[uid][var]):
                    states_type.append(var)
                    states_name.append(var + '_' + uid[0] + str(i + 1))

        if uid == "predicted_observations":
            for var in data[uid]['variables']:
                for i in range(0, data[uid]['horizon']):
                    states_type.append(var)
                    states_name.append(var + '_' + uid[0] + str(i + 1))
    states_type.sort()
    states_name.sort(key=natural_keys)

    min_values = []
    max_values = []
    for i in states_type:
        min_values.append(rescaling_table[i]["min"])
        max_values.append(rescaling_table[i]["max"])

    return states_type, states_name, min_values, max_values


def get_daily_schedule_by_band(F1: float, F2: float, F3: float, timestep_per_hour=1):
    """
    This method can be used to generate schedules according to Italian ToU tariffs.

    :param F1: high price.
    :param F2: medium price.
    :param F3: low price.
    :param timestep_per_hour: number of time steps per hour.
    :return: three numpy arrays. Schedules of weekdays, saturdays and sundays
    """

    weekday = np.concatenate((np.repeat(F3, 7 * timestep_per_hour), np.repeat(F2, 1 * timestep_per_hour),
                              np.repeat(F1, 11 * timestep_per_hour), np.repeat(F2, 4 * timestep_per_hour),
                              np.repeat(F3, 1 * timestep_per_hour)))
    saturday = np.concatenate((np.repeat(F3, 7 * timestep_per_hour), np.repeat(F2, 16 * timestep_per_hour),
                               np.repeat(F3, 1 * timestep_per_hour)))
    sunday = np.repeat(F3, 24 * timestep_per_hour)
    return weekday, saturday, sunday


def get_electricity_price_schedule(F1: float, F2: float, F3: float,
                                   daystart: int, numberofdays: int, timestep_per_hour=1, shift_in_hours=0):
    """

    This method is used to generate a schedule according to Italian ToU tariffs for a specific number of days

    :param F1: high price.
    :param F2: medium price.
    :param F3: low price.
    :param daystart: daytype of the first day of the given period.
    :param numberofdays: total number of days.
    :param timestep_per_hour: number of timestep per hour.
    :param shift_in_hours: (optional) number of hours to forward shift the schedule.
    :return:
    """

    wd_scd, sat_scd, sun_scd = get_daily_schedule_by_band(F1=F1,
                                                          F2=F2,
                                                          F3=F3,
                                                          timestep_per_hour=timestep_per_hour)
    numberofweeks = int(np.floor(numberofdays / 7))
    daysleft = numberofdays - numberofweeks * 7

    if daystart == 7:
        week = np.concatenate((sun_scd, np.tile(wd_scd, 5), sat_scd))
    elif daystart == 6:
        week = np.concatenate((sat_scd, sun_scd, np.tile(wd_scd, 5)))
    else:
        week = np.concatenate((np.tile(wd_scd, 6 - daystart), sat_scd, sun_scd, np.tile(wd_scd, (5 - (6 - daystart)))))
    schedule = np.tile(week, numberofweeks)

    finalperiod = week[0:(daysleft * int(len(week) / 7))]

    schedule = np.concatenate((schedule, finalperiod))
    schedule = np.roll(schedule, shift_in_hours * timestep_per_hour)
    return schedule


def get_eplus_action_encoding(action):
    """

    Return the energy plus encodings for the control action selected by the control strategy.

    :param action: float between -1 and 1.
    :return: list of eplus commands.
    """

    if action > 0:
        eplus_commands = [1, 1, 1, 1, 0]
    elif action < 0:
        eplus_commands = [0, 1, 0, 0, 1]
    else:
        eplus_commands = [0, 1, 1, 0, 0]

    return eplus_commands


def evaluate_perfect_predictions(data, variable_name, horizhon, n_days):
    """

    :param data:
    :param variable_name:
    :param horizhon:
    :param n_days:
    :return:
    """

    data['hour'] = np.ceil(data.time)

    data['day_index'] = np.repeat(range(1, n_days + 1), 12 * 24)

    data_hour = data.groupby(['day_index', 'hour'], as_index=False)[variable_name].mean()

    var_prediction = []
    for i in range(1, horizhon + 1):
        var_prediction.append(data_hour[variable_name].shift(-i))

    var_prediction = pd.concat(var_prediction, axis=1)

    var_prediction.dropna(inplace=True)

    return var_prediction


def calculate_tank_soc(temperature, min_temperature, max_temperature):
    """

    :param temperature:
    :param min_temperature:
    :param max_temperature:
    :return:
    """

    soc = 1 - (temperature - min_temperature) / (max_temperature - min_temperature)

    soc = np.clip(soc, 0, 1)

    return soc


def order_state_variables(env, observation, cooling_load_predictions, electricity_price_predictions,
                          pv_power_generation_predictions, horizon, step):
    """

    :param env:
    :param observation:
    :param cooling_load_predictions:
    :param electricity_price_predictions:
    :param horizon:
    :param step:
    :return:
    """

    state_variables_mask = pd.DataFrame(index=range(0, len(env.state_names)), columns=['variable'])
    state_variables_mask['variable'] = env.state_names
    observation_df = pd.DataFrame(index=range(0, len(observation)), columns=['variable', 'value'])
    observation_df['variable'] = ['outdoor_air_temperature_0', 'cooling_load_0', 'electricity_price_0',
                                  'storage_soc_0', 'storage_soc_l1', 'storage_soc_l2', 'storage_soc_l3',
                                  'storage_soc_l4', 'pv_power_generation_0', 'auxiliary_load_0', 'battery_soc_0']
    observation_df['value'] = observation

    cooling_load_predictions_df = pd.DataFrame(index=range(0, horizon), columns=['variable', 'value'])
    cooling_load_predictions_df['variable'] = cooling_load_predictions.columns[0:horizon]
    cooling_load_predictions_df['value'] = cooling_load_predictions.iloc[step, 0:horizon].values

    electricity_price_predictions_df = pd.DataFrame(index=range(0, horizon), columns=['variable', 'value'])
    electricity_price_predictions_df['variable'] = electricity_price_predictions.columns[0:horizon]
    electricity_price_predictions_df['value'] = electricity_price_predictions.iloc[step, 0:horizon].values

    pv_power_predictions_df = pd.DataFrame(index=range(0, horizon), columns=['variable', 'value'])
    pv_power_predictions_df['variable'] = pv_power_generation_predictions.columns[0:horizon]
    pv_power_predictions_df['value'] = pv_power_generation_predictions.iloc[step, 0:horizon].values

    state_variables = pd.concat([observation_df, electricity_price_predictions_df, cooling_load_predictions_df,
                                 pv_power_predictions_df])

    state_variables_mask = state_variables_mask.merge(state_variables)

    output = tuple(state_variables_mask['value'].values)

    return output


def min_max_scaling(mat, mins, maxs, min_val, max_val):
    """

    :param mat:
    :param mins:
    :param maxs:
    :param min_val:
    :param max_val:
    :return:
    """

    scaled_mat = (max_val - min_val) * ((mat - mins) / (maxs - mins)) + min_val
    return scaled_mat


def set_occupancy_schedule(schedule, index):
    if index == 0:
        schedule['Field_3'] = 'Until 8:30'
        schedule['Field_5'] = 'Until 18:00'
        schedule['Field_10'] = 'Until 8:30'
        schedule['Field_12'] = 'Until 18:00'
        schedule['Field_13'] = 0
        schedule['Field_18'] = 'Until 8:30'
        schedule['Field_20'] = 'Until 18:00'
        schedule['Field_25'] = 'Until 8:30'
        schedule['Field_27'] = 'Until 18:00'
        schedule['Field_28'] = 0
        schedule['Field_33'] = 'Until 8:30'
        schedule['Field_35'] = 'Until 18:00'
        schedule['Field_40'] = 'Until 8:30'
        schedule['Field_42'] = 'Until 18:00'
        schedule['Field_43'] = 0

    elif index == 1:
        schedule['Field_3'] = 'Until 8:30'
        schedule['Field_5'] = 'Until 18:00'
        schedule['Field_10'] = 'Until 8:30'
        schedule['Field_12'] = 'Until 18:00'
        schedule['Field_13'] = 0
        schedule['Field_18'] = 'Until 8:00'
        schedule['Field_20'] = 'Until 19:00'
        schedule['Field_25'] = 'Until 8:30'
        schedule['Field_27'] = 'Until 18:00'
        schedule['Field_28'] = 0
        schedule['Field_33'] = 'Until 8:30'
        schedule['Field_35'] = 'Until 18:00'
        schedule['Field_40'] = 'Until 8:30'
        schedule['Field_42'] = 'Until 18:00'
        schedule['Field_43'] = 0

    elif index == 2:
        schedule['Field_3'] = 'Until 8:30'
        schedule['Field_5'] = 'Until 18:00'
        schedule['Field_10'] = 'Until 8:30'
        schedule['Field_12'] = 'Until 18:00'
        schedule['Field_13'] = 0
        schedule['Field_18'] = 'Until 8:30'
        schedule['Field_20'] = 'Until 18:00'
        schedule['Field_25'] = 'Until 8:30'
        schedule['Field_27'] = 'Until 18:00'
        schedule['Field_28'] = 0
        schedule['Field_33'] = 'Until 8:00'
        schedule['Field_35'] = 'Until 19:00'
        schedule['Field_40'] = 'Until 8:30'
        schedule['Field_42'] = 'Until 18:00'
        schedule['Field_43'] = 0

    elif index == 3:
        schedule['Field_3'] = 'Until 8:30'
        schedule['Field_5'] = 'Until 18:00'
        schedule['Field_10'] = 'Until 8:30'
        schedule['Field_12'] = 'Until 18:00'
        schedule['Field_13'] = 0
        schedule['Field_18'] = 'Until 8:30'
        schedule['Field_20'] = 'Until 18:00'
        schedule['Field_25'] = 'Until 8:30'
        schedule['Field_27'] = 'Until 14:00'
        schedule['Field_28'] = 1
        schedule['Field_33'] = 'Until 8:00'
        schedule['Field_35'] = 'Until 19:00'
        schedule['Field_40'] = 'Until 8:30'
        schedule['Field_42'] = 'Until 14:00'
        schedule['Field_43'] = 1

    return schedule