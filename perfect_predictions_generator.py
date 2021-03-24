import pandas as pd
from utils import evaluate_perfect_predictions


print('Generating perfect predictions...')
forcing_variables = pd.read_csv('supportFiles\\forcing_variables.csv')
vars = ['cooling_load', 'electricity_price', 'pv_power_generation']

for var in vars:
    variable_to_predict = var

    prediction = evaluate_perfect_predictions(data=forcing_variables,
                                              variable_name=variable_to_predict,
                                              horizhon=48,
                                              n_days=92)
    prediction.columns = [variable_to_predict + f'_p{i}' for i in range(1, 49)]
    prediction.to_csv('supportFiles\\prediction-{}_perfect.csv'.format(variable_to_predict), index=False,
                      float_format='%g')
print('Perfect Predictions Generated Successfully!')
