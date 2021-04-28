import pandas as pd
import matplotlib.pyplot as plt

# Load data
directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'
directory_plot = 'C:\\Users\\agall\\OneDrive\\Desktop\\Plots\\Area/'

data_ep = pd.read_csv(directory + 'test_09/configuration_008/episode_25.csv', sep=';', decimal=',', index_col=0)
data_base = pd.read_csv(directory + 'test_09/configuration_008/baseline.csv', sep=';', decimal=',', index_col=0)

assex = data_base.index.to_numpy()
fig, ((ax_electricity_base, ax_electricity_ep), (ax_heat_base, ax_heat_ep)) = plt.subplots(2, 2, figsize=(18, 6),
                                                                                           tight_layout=True,
                                                                                           sharex=True, sharey='row')
fig.suptitle('Storage tank SoC at 4800 Wh and volume 10 m^3')

ax_electricity_base.stackplot(assex, data_base['PV energy to building [J]'].mul(-1 / 3600000),
                              data_base['battery energy to building [J]'].mul(-1 / 3600000),
                              data_base['grid energy to building [J]'].mul(-1 / 3600000),
                              colors=('lawngreen', 'forestgreen', 'darkgreen'),
                              labels=('PV to building', 'Battery discharge', 'Grid to building'),
                              alpha=1, ec='black')
ax_electricity_base.stackplot(assex, data_base['Building load [J]'].mul(1 / 3600000),
                              data_base['PV energy to battery [J]'].mul(1 / 3600000),
                              data_base['PV energy to grid [J]'].mul(1 / 3600000),
                              colors=('cornflowerblue', 'mediumslateblue', 'blue'),
                              labels=('Building load', 'Battery charge', 'PV to grid'),
                              alpha=1, ec='black')
# ax_electricity_base.set_ylabel('Power [kW]', fontsize='large')

# ax.stackplot(assex, data['battery energy to building [J]'].mul(-1/3600000), labels='Battery')
ax2_electricity_base = ax_electricity_base.twinx()
ax2_electricity_base.plot(assex, data_base['Battery soc'], label='BESS SoC', color='gold')
# ax2.plot(assex, data['Tank SOC'], label='Storage SoC', color='mediumslateblue')

ax_electricity_base.set_title('Electricity power Rule-Based')
ax_electricity_base.set_ylim([-4, 4])

ax_heat_base.stackplot(assex, data_base['STORAGETANK:Chilled Water Thermal Storage Use Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                       data_base['CHILLER:Chiller Evaporator Cooling Energy [J](TimeStep)'].mul(-1 / 3600000),
                       colors=('limegreen', 'firebrick'), labels=('Tank discharge', 'Chiller'),
                       alpha=1, ec='black')
ax_heat_base.stackplot(assex, data_base['STORAGETANK:Chilled Water Thermal Storage Source Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                       data_base['Cooling load [W]'].mul(-1 / 1000),
                       colors=('darkgreen', 'mediumslateblue'), labels=('Tank charge', 'Cooling demand'),
                       alpha=1, ec='black')
# ax_heat_base.set_ylabel('Power [kW]', fontsize='large', loc='center')

# ax.stackplot(assex, data['battery energy to building [J]'].mul(-1/3600000), labels='Battery')

ax2_heat_base = ax_heat_base.twinx()
# ax2_heat.plot(assex, data['Battery soc'], label='BESS SoC', color='gold')
ax2_heat_base.plot(assex, data_base['Tank SOC'], label='Storage SoC', color='darkblue')

ax_heat_base.set_title('Thermal power Rule-Based')
day = range(0, 2160, 24)
ax_heat_base.set_xticks(assex[day])
ax_heat_base.set_xticklabels(assex[day], rotation=90)
ax_heat_base.set_ylim([-10, 10])


ax_electricity_ep.stackplot(assex, data_ep['PV energy to building [J]'].mul(-1 / 3600000),
                            data_ep['battery energy to building [J]'].mul(-1 / 3600000),
                            data_ep['grid energy to building [J]'].mul(-1 / 3600000),
                            colors=('lawngreen', 'forestgreen', 'darkgreen'),
                            labels=('PV to building', 'Battery discharge', 'Grid to building'),
                            alpha=1, ec='black')
ax_electricity_ep.stackplot(assex, data_ep['Building load [J]'].mul(1 / 3600000),
                            data_ep['PV energy to battery [J]'].mul(1 / 3600000),
                            data_ep['PV energy to grid [J]'].mul(1 / 3600000),
                            colors=('cornflowerblue', 'mediumslateblue', 'blue'),
                            labels=('Building load', 'Battery charge', 'PV to grid'),
                            alpha=1, ec='black')

# ax.stackplot(assex, data['battery energy to building [J]'].mul(-1/3600000), labels='Battery')
ax2_electricity_ep = ax_electricity_ep.twinx()
ax2_electricity_ep.plot(assex, data_ep['Battery soc'], label='BESS SoC', color='gold')
# ax2.plot(assex, data['Tank SOC'], label='Storage SoC', color='mediumslateblue')
# ax2_electricity_ep.set_ylabel('State of Charge', fontsize='large')

ax_electricity_ep.set_title('Electricity power Reinforcement Learning')
ax_electricity_ep.legend(bbox_to_anchor=(1.05, 0.90, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)

ax_heat_ep.stackplot(assex, data_ep['STORAGETANK:Chilled Water Thermal Storage Use Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                     data_ep['CHILLER:Chiller Evaporator Cooling Energy [J](TimeStep)'].mul(-1 / 3600000),
                     colors=('limegreen', 'firebrick'), labels=('Tank discharge', 'Chiller'),
                     alpha=1, ec='black')
ax_heat_ep.stackplot(assex, data_ep['STORAGETANK:Chilled Water Thermal Storage Source Side Heat Transfer Energy [J](TimeStep)'].mul(-1 / 3600000),
                     data_ep['Cooling load [W]'].mul(-1 / 1000),
                     colors=('darkgreen', 'mediumslateblue'), labels=('Tank charge', 'Cooling demand'),
                     alpha=1, ec='black')

# ax.stackplot(assex, data['battery energy to building [J]'].mul(-1/3600000), labels='Battery')

ax2_heat_ep = ax_heat_ep.twinx()
# ax2_heat.plot(assex, data['Battery soc'], label='BESS SoC', color='gold')
ax2_heat_ep.plot(assex, data_ep['Tank SOC'], label='Storage SoC', color='darkblue')
# ax2_heat_ep.set_ylabel('State of Charge', fontsize='large')

ax_heat_ep.set_title('Thermal power Reinforcement Learning')
ax_heat_ep.legend(bbox_to_anchor=(1.05, 0.90, 0.5, .102), loc='upper left', ncol=1, borderaxespad=0.)
day = range(23, 2160, 24)
ax_heat_ep.set_xticks(assex[day])
ax_heat_ep.set_xticklabels(assex[day], rotation=90)

fig.text(0.0, 0.5, 'Power [kW]', ha='center', va='center', rotation='vertical', fontsize='large')
plt.xlim([' 08/02  21:00:00', ' 08/04  23:00:00'])
# plt.savefig(directory_plot + 'RL vs RB thermal and electricity.png')
