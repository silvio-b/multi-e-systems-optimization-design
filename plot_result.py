import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Access data
directory_plot = 'C:\\Users\\agall\\OneDrive\\Desktop\\Plots\\'
directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'
file = 'test_09.csv'

data = pd.read_csv(directory + file, sep=';', decimal=',')

# Elenco plot
# A barre: SS, SC baseline e proposed per tank, battery
# A barre: Energy consumption, energy from battery baseline e proposed per tank, battery

########################################################################################################################

configurations = [3, 4, 5]
data_self = data[['RL Self-Sufficiency', 'RL Self-Consumption', 'RBC Self-Sufficiency', 'RBC Self-Consumption']]
data_cost = data[['RL Energy cost [Euro]', 'RBC Energy cost [Euro]']]
labels = ['10 m^3', '8 m^3', '6 m^3']

_, ax1 = plt.subplots()

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

ax1.bar(x-3/2*width, data_self.loc[configurations]['RL Self-Sufficiency'], width, color='steelblue', label='RL-SS')
ax1.bar(x-width/2, data_self.loc[configurations]['RBC Self-Sufficiency'], width, color='skyblue', label='RBC-SS')
ax1.bar(x+width/2, data_self.loc[configurations]['RL Self-Consumption'], width, color='orangered', label='RL-SC')
ax1.bar(x+3/2*width, data_self.loc[configurations]['RBC Self-Consumption'], width, color='lightsalmon', label='RBC-SC')

ax2 = ax1.twinx()
ax2.scatter(x, data_cost.loc[configurations]['RL Energy cost [Euro]'], color='k', label='RL-Energy cost', marker='o')
ax2.scatter(x, data_cost.loc[configurations]['RBC Energy cost [Euro]'], color='g', label='RBC-Energy cost', marker='o')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylim(0.0, 1.0)
ax1.set_xlabel('Tank volume')
ax1.set_title('SS and SC for RB and RL at 4800 Wh and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')

ax2.set_ylim(10, 20.0)
ax2.set_ylabel('Euro')
ax2.legend(loc='lower right')
plt.savefig(directory_plot + 'SS and SC for RB and RL at 4800 Wh and seed 137 with energy cost.png')
########################################################################################################################
configurations = [22, 4, 13]
labels = ['2400 Wh', '4800 Wh', '7200 Wh']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

_, ax1 = plt.subplots()

ax1.bar(x-3*width/2, data_self.loc[configurations]['RL Self-Sufficiency'], width, color='steelblue', label='RL-SS')
ax1.bar(x-width/2, data_self.loc[configurations]['RBC Self-Sufficiency'], width, color='skyblue', label='RBC-SS')
ax1.bar(x+width/2, data_self.loc[configurations]['RL Self-Consumption'], width, color='orangered', label='RL-SC')
ax1.bar(x+3*width/2, data_self.loc[configurations]['RBC Self-Consumption'], width, color='lightsalmon', label='RBC-SC')

ax2 = ax1.twinx()
ax2.scatter(x, data_cost.loc[configurations]['RL Energy cost [Euro]'], color='k', label='RL-Energy cost', marker='o')
ax2.scatter(x, data_cost.loc[configurations]['RBC Energy cost [Euro]'], color='g', label='RBC-Energy cost', marker='o')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel('Battery capacity')
ax1.set_ylim(0.0, 1.0)
ax1.set_title('SS and SC for RB and RL at 8 m^3 and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')

ax2.set_ylim(10, 25.0)
ax2.set_ylabel('Euro')
ax2.legend(loc='lower right')
plt.savefig(directory_plot + 'SS and SC for RB and RL at 8 m^3 and seed 137 with energy cost.png')
########################################################################################################################
configurations = [3, 4, 5]
data_energy = data[['RL Energy consumption [kWh]', 'RL Energy battery [kWh]', 'RBC Energy consumption [kWh]', 'RBC Energy battery [kWh]']]
labels = ['10 m^3', '8 m^3', '6 m^3']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

_, ax1 = plt.subplots()

ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width, color='steelblue', label='RL-Consumption')
ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width, color='skyblue', label='RL-Battery')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width, color='orangered', label='RBC-Consumption')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width, color='lightsalmon', label='RBC-Battery')

ax2 = ax1.twinx()
ax2.scatter(x, data_cost.loc[configurations]['RL Energy cost [Euro]'], color='k', label='RL-Energy cost', marker='o')
ax2.scatter(x, data_cost.loc[configurations]['RBC Energy cost [Euro]'], color='g', label='RBC-Energy cost', marker='o')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel('Tank volume')
ax1.set_title('Energy consumption and from battery for RB and RL at 4800 Wh and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')
ax1.set_ylim(0.0, 1100.0)

ax2.set_ylim(10, 25.0)
ax2.set_ylabel('Euro')
ax2.legend(loc='upper right')
plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 4800 Wh and seed 137 with energy cost.png')
########################################################################################################################
configurations = [22, 4, 13]
labels = ['2400 Wh', '4800 Wh', '7200 Wh']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

_, ax1 = plt.subplots()

ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width, color='steelblue', label='RL-Consumption')
ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width, color='skyblue', label='RL-Battery')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width, color='orangered', label='RBC-Consumption')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width, color='lightsalmon', label='RBC-Battery')

ax2 = ax1.twinx()
ax2.scatter(x, data_cost.loc[configurations]['RL Energy cost [Euro]'], color='k', label='RL-Energy cost', marker='o')
ax2.scatter(x, data_cost.loc[configurations]['RBC Energy cost [Euro]'], color='g', label='RBC-Energy cost', marker='o')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel('Battery capacity')
ax1.set_title('Energy consumption and from battery for RB and RL at 8 m^3 and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')
ax1.set_ylim(0.0, 1100.0)

ax2.set_ylim(10, 25.0)
ax2.set_ylabel('Euro')
ax2.legend(loc='lower right')
plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 8 m^3 and seed 137 with energy cost.png')
########################################################################################################################
########################################################################################################################
# Same as before but with COP instead of cost

configurations = [3, 4, 5]
data_self = data[['RL Self-Sufficiency', 'RL Self-Consumption', 'RBC Self-Sufficiency', 'RBC Self-Consumption']]
data_cop = data[['RL COP global', 'RBC COP global']]
labels = ['10 m^3', '8 m^3', '6 m^3']

_, ax1 = plt.subplots()

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

ax1.bar(x-3/2*width, data_self.loc[configurations]['RL Self-Sufficiency'], width, color='steelblue', label='RL-SS')
ax1.bar(x-width/2, data_self.loc[configurations]['RBC Self-Sufficiency'], width, color='skyblue', label='RBC-SS')
ax1.bar(x+width/2, data_self.loc[configurations]['RL Self-Consumption'], width, color='orangered', label='RL-SC')
ax1.bar(x+3/2*width, data_self.loc[configurations]['RBC Self-Consumption'], width, color='lightsalmon', label='RBC-SC')

ax2 = ax1.twinx()
ax2.scatter(x, data_cop.loc[configurations]['RL COP global'], color='k', label='RL-COP global', marker='o')
ax2.scatter(x, data_cop.loc[configurations]['RBC COP global'], color='g', label='RBC-COP global', marker='o')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylim(0.0, 1.0)
ax1.set_xlabel('Tank volume')
ax1.set_title('SS and SC for RB and RL at 4800 Wh and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')

ax2.set_ylim(0, 5.0)
ax2.legend(loc='lower right')
plt.savefig(directory_plot + 'SS and SC for RB and RL at 4800 Wh and seed 137 with COP.png')
########################################################################################################################
configurations = [22, 4, 13]
labels = ['2400 Wh', '4800 Wh', '7200 Wh']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

_, ax1 = plt.subplots()

ax1.bar(x-3*width/2, data_self.loc[configurations]['RL Self-Sufficiency'], width, color='steelblue', label='RL-SS')
ax1.bar(x-width/2, data_self.loc[configurations]['RBC Self-Sufficiency'], width, color='skyblue', label='RBC-SS')
ax1.bar(x+width/2, data_self.loc[configurations]['RL Self-Consumption'], width, color='orangered', label='RL-SC')
ax1.bar(x+3*width/2, data_self.loc[configurations]['RBC Self-Consumption'], width, color='lightsalmon', label='RBC-SC')

ax2 = ax1.twinx()
ax2.scatter(x, data_cop.loc[configurations]['RL COP global'], color='k', label='RL-COP global', marker='o')
ax2.scatter(x, data_cop.loc[configurations]['RBC COP global'], color='g', label='RBC-COP global', marker='o')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel('Battery capacity')
ax1.set_ylim(0.0, 1.0)
ax1.set_title('SS and SC for RB and RL at 8 m^3 and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')

ax2.set_ylim(0.0, 5.0)
ax2.legend(loc='lower right')
plt.savefig(directory_plot + 'SS and SC for RB and RL at 8 m^3 and seed 137 with CoP.png')
########################################################################################################################
configurations = [3, 4, 5]
data_energy = data[['RL Energy consumption [kWh]', 'RL Energy battery [kWh]', 'RBC Energy consumption [kWh]', 'RBC Energy battery [kWh]']]
labels = ['10 m^3', '8 m^3', '6 m^3']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

_, ax1 = plt.subplots()

ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width, color='steelblue', label='RL-Consumption')
ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width, color='skyblue', label='RL-Battery')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width, color='orangered', label='RBC-Consumption')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width, color='lightsalmon', label='RBC-Battery')

ax2 = ax1.twinx()
ax2.scatter(x, data_cop.loc[configurations]['RL COP global'], color='k', label='RL-COP global', marker='o')
ax2.scatter(x, data_cop.loc[configurations]['RBC COP global'], color='g', label='RBC-COP global', marker='o')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel('Tank volume')
ax1.set_title('Energy consumption and from battery for RB and RL at 4800 Wh and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')
ax1.set_ylim(0.0, 1300.0)

ax2.set_ylim(0.0, 5.0)
ax2.legend(loc='upper right')
plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 4800 Wh and seed 137 with CoP.png')
########################################################################################################################
configurations = [22, 4, 13]
labels = ['2400 Wh', '4800 Wh', '7200 Wh']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

_, ax1 = plt.subplots()

ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width, color='steelblue', label='RL-Consumption')
ax1.bar(x-width/2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width, color='skyblue', label='RL-Battery')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width, color='orangered', label='RBC-Consumption')
ax1.bar(x+width/2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width, color='lightsalmon', label='RBC-Battery')

ax2 = ax1.twinx()
ax2.scatter(x, data_cop.loc[configurations]['RL COP global'], color='k', label='RL-COP global', marker='o')
ax2.scatter(x, data_cop.loc[configurations]['RBC COP global'], color='g', label='RBC-COP global', marker='o')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel('Battery capacity')
ax1.set_title('Energy consumption and from battery for RB and RL at 8 m^3 and seed 137')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(loc='lower left')
ax1.set_ylim(0.0, 1300.0)

ax2.set_ylim(0.0, 5.0)
ax2.legend(loc='lower right')
plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 8 m^3 and seed 137 with CoP.png')
########################################################################################################################
