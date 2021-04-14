import pandas as pd
import numpy as np
from utils import get_electricity_price_schedule

F1 = 0.3
F2 = 0.165
F3 = 0.03

simulation_time_step = 1

electricity_price_schedule = get_electricity_price_schedule(F1=F1,
                                                            F2=F2,
                                                            F3=F3,
                                                            daystart=2,
                                                            numberofdays=92,
                                                            timestep_per_hour=simulation_time_step,
                                                            shift_in_hours=0)

electricity_price_schedule = pd.DataFrame(electricity_price_schedule)
electricity_price_schedule.to_csv('supportFiles/electricity_price_schedule_hour.csv', index=False, header=False,
                                  float_format='%g')
