import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Access data
directory_plot = 'C:\\Users\\agall\\OneDrive\\Desktop\\Plots\\Bar/Average/'
directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'
file = 'average_seeds.csv'

data = pd.read_csv(directory + file, sep=';', decimal=',')

cost_rl = data['RL Energy cost [Euro]']
cost_rb = data['RBC Energy cost [Euro]']

energy_rl = data['RL Energy consumption [kWh]']
energy_rb = data['RBC Energy consumption [kWh]']

# setup the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})

# Plot data
_x = np.arange(3)
_y = np.arange(3)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

cost = (cost_rb - cost_rl) / cost_rb * 100
energy = (energy_rb - energy_rl) / energy_rb * 100
bottom = np.zeros_like(cost)
width = depth = 0.5

ax1.bar3d(x-width/2, y-width/2, bottom, width, depth, cost, shade=True, ec='black',
          color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen', 'limegreen', 'limegreen', 'firebrick', 'firebrick', 'firebrick'])

ax1.set_title(r'$\Delta$ Energy cost')

ax1.set_xticks(_x)
ax1.set_xticklabels(['2400', '4800', '7200'])
ax1.set_yticks(_y)
ax1.set_yticklabels(['6', '8', '10'])

ax1.set_xlabel('BESS capacity [Wh]')
ax1.set_ylabel('Tank volume [$m^3$]')
ax1.set_zlabel('%')

ax2.bar3d(x-width/2, y-width/2, bottom, width, depth, energy, shade=True, ec='black',
          color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen', 'limegreen', 'limegreen', 'firebrick', 'firebrick', 'firebrick'])

ax2.set_title(r'$\Delta$ Energy consumption')

ax2.set_xticks(_x)
ax2.set_xticklabels(['2400', '4800', '7200'])
ax2.set_yticks(_y)
ax2.set_yticklabels(['6', '8', '10'])

ax2.set_xlabel('BESS capacity [Wh]')
ax2.set_ylabel('Tank volume [$m^3$]')
ax2.set_zlabel('%')

plt.savefig(directory_plot + 'bar3d_percento.png')
plt.close()

# Energy consumption
# setup the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d', 'sharez': ax1})

plt.suptitle('Energy consumption')

ax1.bar3d(x-width/2, y-width/2, bottom, width, depth, energy_rl, shade=True, ec='black',
          color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen', 'limegreen', 'limegreen', 'firebrick', 'firebrick', 'firebrick'])

ax1.set_title('Advanced control strategy')

ax1.set_xticks(_x)
ax1.set_xticklabels(['2400', '4800', '7200'])
ax1.set_yticks(_y)
ax1.set_yticklabels(['6', '8', '10'])

ax1.set_xlabel('BESS capacity [Wh]')
ax1.set_ylabel('Tank volume [$m^3$]')
ax1.set_zlabel('Energy consumption [kWh]')

ax2.bar3d(x-width/2, y-width/2, bottom, width, depth, energy_rb, shade=True, ec='black',
          color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen', 'limegreen', 'limegreen', 'firebrick', 'firebrick', 'firebrick'])

ax2.set_title('Baseline')

ax2.set_xticks(_x)
ax2.set_xticklabels(['2400', '4800', '7200'])
ax2.set_yticks(_y)
ax2.set_yticklabels(['6', '8', '10'])

ax2.set_xlabel('BESS capacity [Wh]')
ax2.set_ylabel('Tank volume [$m^3$]')
ax2.set_zlabel('Energy consumption [kWh]')

plt.savefig(directory_plot + 'bar3d_energy.png')
plt.close()

# Energy cost
# setup the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d', 'sharez': ax1})

plt.suptitle('Energy cost')

ax1.bar3d(x-width/2, y-width/2, bottom, width, depth, cost_rl, shade=True, ec='black',
          color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen', 'limegreen', 'limegreen', 'firebrick', 'firebrick', 'firebrick'])

ax1.set_title('Advanced control strategy')

ax1.set_xticks(_x)
ax1.set_xticklabels(['2400', '4800', '7200'])
ax1.set_yticks(_y)
ax1.set_yticklabels(['6', '8', '10'])

ax1.set_xlabel('BESS capacity [Wh]')
ax1.set_ylabel('Tank volume [$m^3$]')
ax1.set_zlabel('Energy cost [€]')

ax2.bar3d(x-width/2, y-width/2, bottom, width, depth, cost_rb, shade=True, ec='black',
          color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen', 'limegreen', 'limegreen', 'firebrick', 'firebrick', 'firebrick'])

ax2.set_title('Baseline')

ax2.set_xticks(_x)
ax2.set_xticklabels(['2400', '4800', '7200'])
ax2.set_yticks(_y)
ax2.set_yticklabels(['6', '8', '10'])

ax2.set_xlabel('BESS capacity [Wh]')
ax2.set_ylabel('Tank volume [$m^3$]')
ax2.set_zlabel('Energy cost [€]')

plt.savefig(directory_plot + 'bar3d_cost.png')
# plt.show()