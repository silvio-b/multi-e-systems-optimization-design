import pandas as pd
from utils import evaluate_perfect_predictions
occupancy_schedule_index = 1

print('Generating perfect predictions...')
forcing_variables = pd.read_csv('data\\forcing_variables_occ{}.csv'.format(occupancy_schedule_index))
vars = ['cooling_load', 'electricity_price']

for var in vars:
    variable_to_predict = var

    prediction = evaluate_perfect_predictions(data=forcing_variables,
                                              variable_name=variable_to_predict,
                                              horizhon=48,
                                              n_days=92,
                                              ep_timestep=1)
    prediction.columns = [variable_to_predict + f'_p{i}' for i in range(1, 49)]
    prediction.to_csv('supportFiles\\prediction-{}_perfect_occ{}.csv'.format(variable_to_predict,
                                                                             occupancy_schedule_index), index=False,
                      float_format='%g')
print('Perfect Predictions Generated Successfully!')
#
# prediction = evaluate_perfect_predictions(data=forcing_variables,
#                                           variable_name='pv_power_generation',
#                                           horizhon=48,
#                                           n_days=92,
#                                           ep_timestep=1)
#
# prediction.columns = ['pv_power_generation' + f'_p{i}' for i in range(1, 49)]
# prediction.to_csv('supportFiles\\prediction-{}_perfect_2000.csv'.format('pv_power_generation'), index=False,
#                   float_format='%g')
