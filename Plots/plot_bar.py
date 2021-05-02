import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'
directory_plot = 'C:\\Users\\agall\\OneDrive\\Desktop\\Plots\\'

# data_base = pd.read_csv(directory + 'test_09/configuration_008/episode_25.csv', sep=';', decimal=',', index_col=0)
data_base = pd.read_csv(directory + 'test_09/configuration_008/baseline.csv', sep=';', decimal=',', index_col=0)

width = 0.35
day = range(23, 2160, 24)

assex = np.arange(len(data_base.index.to_numpy()))
assex_labels = data_base.index.to_numpy()
# fig, (ax_electricity_base, ax_heat_base) = plt.subplots(2, 1, figsize=(18, 6), tight_layout=True, sharex=True)

fig, ax_electricity_base = plt.subplots(figsize=(18, 6), tight_layout=True, sharex=True, sharey='row')

pv_to_building = ax_electricity_base.bar(assex, data_base['PV energy to building [J]'].mul(-1 / 3600000), width,
                        color='palegreen', label='PV to building', alpha=1, ec='black')
pv_to_battery = ax_electricity_base.bar(assex, data_base['PV energy to battery [J]'].mul(-1 / 3600000), width,
                        bottom=data_base['PV energy to building [J]'].mul(-1 / 3600000),
                        color='limegreen', label='PV to battery', alpha=1, ec='black')
battery_discharge = ax_electricity_base.bar(assex, data_base['battery energy to building [J]'].mul(-1 / 3600000), width,
                        bottom=data_base['PV energy to building [J]'].mul(-1 / 3600000) +
                               data_base['PV energy to battery [J]'].mul(-1 / 3600000),
                        color='maroon', label='Battery discharge', alpha=1, ec='black')
grid_to_building = ax_electricity_base.bar(assex, data_base['grid energy to building [J]'].mul(-1 / 3600000), width,
                        bottom=data_base['PV energy to battery [J]'].mul(-1 / 3600000) +
                               data_base['PV energy to building [J]'].mul(-1 / 3600000) +
                               data_base['battery energy to building [J]'].mul(-1 / 3600000),
                        color='blue', label='Grid to building', alpha=1, ec='black')
pv_to_grid = ax_electricity_base.bar(assex, data_base['PV energy to grid [J]'].mul(-1 / 3600000), width,
                        bottom=data_base['PV energy to battery [J]'].mul(-1 / 3600000) +
                               data_base['PV energy to building [J]'].mul(-1 / 3600000) +
                               data_base['battery energy to building [J]'].mul(-1 / 3600000) +
                               data_base['grid energy to building [J]'].mul(-1 / 3600000),
                        color='darkgreen', label='PV to grid', alpha=1, ec='black')


building = ax_electricity_base.bar(assex, data_base['Building load [J]'].mul(1 / 3600000), width,
                        color='cornflowerblue', label='Building load', alpha=1, ec='black')
battery_charge = ax_electricity_base.bar(assex, data_base['PV energy to battery [J]'].mul(1 / 3600000), width,
                        bottom=data_base['Building load [J]'].mul(1 / 3600000),
                        color='tomato', label='Battery charge', alpha=1, ec='black')
sold_to_grid = ax_electricity_base.bar(assex, data_base['PV energy to grid [J]'].mul(1 / 3600000), width,
                        bottom=data_base['Building load [J]'].mul(1 / 3600000) +
                               data_base['PV energy to battery [J]'].mul(1 / 3600000),
                        color='midnightblue', label='PV to grid', alpha=1, ec='black')

ax_electricity_base.set_title('Electrical power')
ax_electricity_base.set_xticks(assex[day])
ax_electricity_base.set_xticklabels(assex_labels[day], rotation=90)
ax_electricity_base.set_xlabel('Date', fontsize=15)
ax_electricity_base.set_ylabel('Energy [kWh]', fontsize=15)
# ax_electricity_base.text()
legend_pv = ax_electricity_base.legend([pv_to_building, pv_to_grid, pv_to_battery], ['PV to building', 'PV to grid', 'PV to battery'],
                                            bbox_to_anchor=(1.05, 0.90, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)
legend_building = ax_electricity_base.legend([building, grid_to_building, sold_to_grid], ['Building load', 'Grid to building', 'Sold to grid'],
                                            bbox_to_anchor=(1.05, 0.55, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)
legend_battery = ax_electricity_base.legend([battery_charge, battery_discharge], ['Battery charge', 'Battery discharge'],
                                            bbox_to_anchor=(1.05, 0.70, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)

plt.gca().add_artist(legend_pv)
plt.gca().add_artist(legend_building)

ax_electricity_base_soc = ax_electricity_base.twinx()
ax_electricity_base_soc.plot(assex, data_base['Battery soc'], label='BESS SoC', color='gold')
ax_electricity_base_soc.legend( bbox_to_anchor=(1.05, 0.35, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)
ax_electricity_base_soc.set_ylabel('State of Charge', fontsize=15)

# , legend_building, legend_battery
plt.xlim([690, 750])
plt.savefig(directory_plot + 'Electrical flows by RB.png')
# plt.close()

fig, ax_heat_base = plt.subplots(figsize=(18, 6), tight_layout=True, sharex=True, sharey='row')

tank_discharge = ax_heat_base.bar(assex, data_base['STORAGETANK:Chilled Water Thermal Storage Use Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                 width, color='limegreen', label='Tank discharge', alpha=1, ec='black')
chiller = ax_heat_base.bar(assex, data_base['CHILLER:Chiller Evaporator Cooling Energy [J](TimeStep)'].mul(-1 / 3600000),
                 width, bottom=data_base['STORAGETANK:Chilled Water Thermal Storage Use Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                 color='firebrick', label='Chiller', alpha=1, ec='black')
tank_charge = ax_heat_base.bar(assex, data_base['STORAGETANK:Chilled Water Thermal Storage Source Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                 width, color='darkgreen', label='Tank charge', alpha=1, ec='black')
cooling_load = ax_heat_base.bar(assex, data_base['Cooling load [W]'].mul(-1 / 1000),
                 width, bottom=data_base['STORAGETANK:Chilled Water Thermal Storage Source Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                 color='mediumslateblue', label='Cooling demand', alpha=1, ec='black')


ax_heat_base.set_title('Thermal power')
ax_heat_base.set_xlabel('Date', fontsize=15)
ax_heat_base.set_ylabel('Energy [kWh]', fontsize=15, loc='center')
ax_heat_base.set_xticks(assex[day])
ax_heat_base.set_xticklabels(assex_labels[day], rotation=90)
legend_tank = ax_heat_base.legend([tank_charge, tank_discharge], ['Tank charge', 'Tank discharge'],
                                  bbox_to_anchor=(1.05, 0.90, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)
ax_heat_base.legend([chiller, cooling_load], ['Chiller', 'Cooling load'],
                    bbox_to_anchor=(1.05, 0.75, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)
plt.gca().add_artist(legend_tank)

ax_heat_base_soc = ax_heat_base.twinx()
ax_heat_base_soc.plot(assex, data_base['Tank SOC'], label='Tank SoC', color='gold')
ax_heat_base_soc.legend(bbox_to_anchor=(1.05, 0.35, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)
ax_heat_base_soc.set_ylabel('State of Charge', fontsize=15)

plt.xlim([690, 750])
plt.savefig(directory_plot + 'Thermal flows by RB.png')
# plt.show()