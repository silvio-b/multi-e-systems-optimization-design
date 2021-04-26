import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Access data
directory_plot = 'C:\\Users\\agall\\OneDrive\\Desktop\\Plots\\Bar/'
directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'
file = 'test_09.csv'

data = pd.read_csv(directory + file, sep=';', decimal=',')

# Elenco plot
# A barre: SS, SC baseline e proposed for all tank and battery at best seed
# A barre: Energy consumption, energy from battery baseline e proposed per tank, battery at best seed

########################################################################################################################
# A barre: SS, SC baseline e proposed for all tank and battery at best seed
data_self = data[['RL Self-Sufficiency', 'RL Self-Consumption', 'RBC Self-Sufficiency', 'RBC Self-Consumption']]

# 3 plots with labels = ['10 m^3', '8 m^3', '6 m^3'] for SS and SC

labels = ['10 m^3', '8 m^3', '6 m^3']
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

# 1) battery: 2400 Wh configuration: 18, 25, 23
configurations = [18, 25, 23]

_, ax = plt.subplots()

ax.bar(x - 3 / 2 * width, data_self.loc[configurations]['RL Self-Sufficiency'], width,
       color='steelblue', label='RL: Self-Sufficiency', ec='black')
ax.bar(x - width / 2, data_self.loc[configurations]['RL Self-Consumption'], width,
       color='skyblue', label='RL: Self-Consumption', ec='black')
ax.bar(x + width / 2, data_self.loc[configurations]['RBC Self-Sufficiency'], width,
       color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
ax.bar(x + 3 / 2 * width, data_self.loc[configurations]['RBC Self-Consumption'], width,
       color='greenyellow', label='RBC: Self-Consumption', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Tank volume')
ax.set_title('SS and SC for RB and RL at 2400 Wh and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')

plt.savefig(directory_plot + 'SS and SC for RB and RL at 2400 Wh and best seed.png')
########################################################################################################################
# 2) battery: 4800 Wh configuration: 3, 7, 8
configurations = [3, 7, 8]

_, ax = plt.subplots()

ax.bar(x - 3 / 2 * width, data_self.loc[configurations]['RL Self-Sufficiency'], width,
       color='steelblue', label='RL: Self-Sufficiency', ec='black')
ax.bar(x - width / 2, data_self.loc[configurations]['RL Self-Consumption'], width,
       color='skyblue', label='RL: Self-Consumption', ec='black')
ax.bar(x + width / 2, data_self.loc[configurations]['RBC Self-Sufficiency'], width,
       color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
ax.bar(x + 3 / 2 * width, data_self.loc[configurations]['RBC Self-Consumption'], width,
       color='greenyellow', label='RBC: Self-Consumption', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Tank volume')
ax.set_title('SS and SC for RB and RL at 4800 Wh and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')

plt.savefig(directory_plot + 'SS and SC for RB and RL at 4800 Wh and best seed.png')
########################################################################################################################
# 3) battery: 7200 Wh configuration: 15, 10, 17
configurations = [15, 10, 17]

_, ax = plt.subplots()

ax.bar(x - 3 / 2 * width, data_self.loc[configurations]['RL Self-Sufficiency'], width,
       color='steelblue', label='RL: Self-Sufficiency', ec='black')
ax.bar(x - width / 2, data_self.loc[configurations]['RL Self-Consumption'], width,
       color='skyblue', label='RL: Self-Consumption', ec='black')
ax.bar(x + width / 2, data_self.loc[configurations]['RBC Self-Sufficiency'], width,
       color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
ax.bar(x + 3 / 2 * width, data_self.loc[configurations]['RBC Self-Consumption'], width,
       color='greenyellow', label='RBC: Self-Consumption', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Tank volume')
ax.set_title('SS and SC for RB and RL at 7200 Wh and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')

plt.savefig(directory_plot + 'SS and SC for RB and RL at 7200 Wh and best seed.png')
########################################################################################################################
# 3 plots with labels = ['2400 Wh', '4800 Wh', '7200 Wh'] for SS and SC
labels = ['2400 Wh', '4800 Wh', '7200 Wh']
x = np.arange(len(labels))  # the label locations

# 1) tank: 10 mc configuration: 18, 3, 15
configurations = [18, 3, 15]

_, ax = plt.subplots()

ax.bar(x - 3 * width / 2, data_self.loc[configurations]['RL Self-Sufficiency'], width,
       color='steelblue', label='RL: Self-Sufficiency', ec='black')
ax.bar(x - width / 2, data_self.loc[configurations]['RL Self-Consumption'], width,
       color='skyblue', label='RL: Self-Consumption', ec='black')
ax.bar(x + width / 2, data_self.loc[configurations]['RBC Self-Sufficiency'], width,
       color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
ax.bar(x + 3 * width / 2, data_self.loc[configurations]['RBC Self-Consumption'], width,
       color='greenyellow', label='RBC: Self-Consumption', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Battery capacity')
ax.set_ylim(0.0, 1.0)
ax.set_title('SS and SC for RB and RL at 10 m^3 and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
plt.savefig(directory_plot + 'SS and SC for RB and RL at 10 m^3 and best seed.png')
########################################################################################################################
# 2) tank: 8 mc configuration: 25, 7, 10
configurations = [25, 7, 10]

_, ax = plt.subplots()

ax.bar(x - 3 * width / 2, data_self.loc[configurations]['RL Self-Sufficiency'], width,
       color='steelblue', label='RL: Self-Sufficiency', ec='black')
ax.bar(x - width / 2, data_self.loc[configurations]['RL Self-Consumption'], width,
       color='skyblue', label='RL: Self-Consumption', ec='black')
ax.bar(x + width / 2, data_self.loc[configurations]['RBC Self-Sufficiency'], width,
       color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
ax.bar(x + 3 * width / 2, data_self.loc[configurations]['RBC Self-Consumption'], width,
       color='greenyellow', label='RBC: Self-Consumption', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Battery capacity')
ax.set_ylim(0.0, 1.0)
ax.set_title('SS and SC for RB and RL at 8 m^3 and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
plt.savefig(directory_plot + 'SS and SC for RB and RL at 8 m^3 and best seed.png')
########################################################################################################################
# 3) tank: 6 mc configuration: 23, 8, 17
configurations = [23, 8, 17]

_, ax = plt.subplots()

ax.bar(x - 3 * width / 2, data_self.loc[configurations]['RL Self-Sufficiency'], width,
       color='steelblue', label='RL: Self-Sufficiency', ec='black')
ax.bar(x - width / 2, data_self.loc[configurations]['RL Self-Consumption'], width,
       color='skyblue', label='RL: Self-Consumption', ec='black')
ax.bar(x + width / 2, data_self.loc[configurations]['RBC Self-Sufficiency'], width,
       color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
ax.bar(x + 3 * width / 2, data_self.loc[configurations]['RBC Self-Consumption'], width,
       color='greenyellow', label='RBC: Self-Consumption', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Battery capacity')
ax.set_ylim(0.0, 1.0)
ax.set_title('SS and SC for RB and RL at 6 m^3 and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
plt.savefig(directory_plot + 'SS and SC for RB and RL at 6 m^3 and best seed.png')
########################################################################################################################

# A barre: Energy consumption, energy from battery baseline e proposed per tank, battery at best seed
data_energy = data[['RL Energy consumption [kWh]', 'RL Energy battery [kWh]',
                    'RBC Energy consumption [kWh]', 'RBC Energy battery [kWh]']]

# 3 plots with labels = ['10 m^3', '8 m^3', '6 m^3'] for energy consumption and from battery
labels = ['10 m^3', '8 m^3', '6 m^3']
x = np.arange(len(labels))  # the label locations

# 1) battery: 2400 Wh configuration: 18, 25, 23
configurations = [18, 25, 23]

_, ax = plt.subplots()

ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width,
       color='steelblue', label='RL-Consumption', ec='black')
ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width,
       color='skyblue', label='RL-Battery', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width,
       color='forestgreen', label='RBC-Consumption', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width,
       color='greenyellow', label='RBC-Battery', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Tank volume')
ax.set_ylabel('Energy [kWh]')
ax.set_title('Energy consumption and from battery for RB and RL at 2400 Wh and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
ax.set_ylim(0.0, 1500.0)

plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 2400 Wh and best seed.png')
########################################################################################################################
# 2) battery: 4800 Wh configuration: 3, 7, 8
configurations = [3, 7, 8]

_, ax = plt.subplots()

ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width,
       color='steelblue', label='RL-Consumption', ec='black')
ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width,
       color='skyblue', label='RL-Battery', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width,
       color='forestgreen', label='RBC-Consumption', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width,
       color='greenyellow', label='RBC-Battery', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Tank volume')
ax.set_ylabel('Energy [kWh]')
ax.set_title('Energy consumption and from battery for RB and RL at 4800 Wh and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
ax.set_ylim(0.0, 1500.0)

plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 4800 Wh and best seed.png')
########################################################################################################################
# 3) battery: 7200 Wh configuration: 15, 10, 17
configurations = [15, 10, 17]

_, ax = plt.subplots()

ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width,
       color='steelblue', label='RL-Consumption', ec='black')
ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width,
       color='skyblue', label='RL-Battery', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width,
       color='forestgreen', label='RBC-Consumption', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width,
       color='greenyellow', label='RBC-Battery', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Tank volume')
ax.set_ylabel('Energy [kWh]')
ax.set_title('Energy consumption and from battery for RB and RL at 7200 Wh and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
ax.set_ylim(0.0, 1500.0)

plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 7200 Wh and best seed.png')
########################################################################################################################
# 3 plots with labels = ['2400 Wh', '4800 Wh', '7200 Wh'] for energy consumption and from battery
labels = ['2400 Wh', '4800 Wh', '7200 Wh']
x = np.arange(len(labels))  # the label locations

# 1) tank: 10 mc configuration: 18, 3, 15
configurations = [18, 3, 15]

_, ax = plt.subplots()

ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width,
       color='steelblue', label='RL-Consumption', ec='black')
ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width,
       color='skyblue', label='RL-Battery', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width,
       color='forestgreen', label='RBC-Consumption', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width,
       color='greenyellow', label='RBC-Battery', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Battery capacity')
ax.set_ylabel('Energy [kWh]')
ax.set_title('Energy consumption and from battery for RB and RL at 10 m^3 and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
ax.set_ylim(0.0, 1500.0)

plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 10 m^3 and best seed.png')
########################################################################################################################
# 2) tank: 8 mc configuration: 25, 7, 10
configurations = [25, 7, 10]

_, ax = plt.subplots()

ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width,
       color='steelblue', label='RL-Consumption', ec='black')
ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width,
       color='skyblue', label='RL-Battery', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width,
       color='forestgreen', label='RBC-Consumption', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width,
       color='greenyellow', label='RBC-Battery', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Battery capacity')
ax.set_ylabel('Energy [kWh]')
ax.set_title('Energy consumption and from battery for RB and RL at 8 m^3 and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
ax.set_ylim(0.0, 1500.0)

plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 8 m^3 and best seed.png')
########################################################################################################################
# 3) tank: 6 mc configuration: 23, 8, 17
configurations = [23, 8, 17]

_, ax = plt.subplots()

ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy consumption [kWh]'], width,
       color='steelblue', label='RL-Consumption', ec='black')
ax.bar(x - width / 2, data_energy.loc[configurations]['RL Energy battery [kWh]'], width,
       color='skyblue', label='RL-Battery', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy consumption [kWh]'], width,
       color='forestgreen', label='RBC-Consumption', ec='black')
ax.bar(x + width / 2, data_energy.loc[configurations]['RBC Energy battery [kWh]'], width,
       color='greenyellow', label='RBC-Battery', ec='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Battery capacity')
ax.set_ylabel('Energy [kWh]')
ax.set_title('Energy consumption and from battery for RB and RL at 6 m^3 and best seed')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right', ncol=1, fontsize='small')
ax.set_ylim(0.0, 1500.0)

plt.savefig(directory_plot + 'Energy consumption and from battery for RB and RL at 6 m^3 and best seed.png')
########################################################################################################################
