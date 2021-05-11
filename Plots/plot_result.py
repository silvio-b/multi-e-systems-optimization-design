import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Access data
directory_plot = 'C:\\Users\\agall\\OneDrive\\Desktop\\Plots\\'
directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'
file = 'best_seeds.csv'

data = pd.read_csv(directory + file, sep=';', decimal=',')

# Elenco plot
# A barre: SS, SC baseline e proposed for all tank and battery at best seed
# A barre: Energy consumption, energy from battery baseline e proposed per tank, battery at best seed


def plot_bar_battery(configurations, labels, width=0.15):

    x = np.arange(len(labels))  # the label locations
    size = data.loc[configurations[0]]['battery_size']
    _, ax = plt.subplots()

    ax.bar(x - 3 / 2 * width, data.loc[configurations]['RL Self-Sufficiency'], width,
           color='steelblue', label='RL: Self-Sufficiency', ec='black')
    ax.bar(x - width / 2, data.loc[configurations]['RL Self-Consumption'], width,
           color='skyblue', label='RL: Self-Consumption', ec='black')
    ax.bar(x + width / 2, data.loc[configurations]['RBC Self-Sufficiency'], width,
           color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
    ax.bar(x + 3 / 2 * width, data.loc[configurations]['RBC Self-Consumption'], width,
           color='greenyellow', label='RBC: Self-Consumption', ec='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Tank volume')
    ax.set_title(f'SS and SC for RB and RL at {size} Wh and best seed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', ncol=1, fontsize='small')

    plt.savefig(directory_plot + f'SS and SC for RB and RL at {size} Wh and best seed.png')
    plt.close()

    _, ax = plt.subplots()

    # ax.bar(x - width / 2, data.loc[configurations]['RL Energy consumption [kWh]'], width,
    #        color='darkblue', label='RL-Consumption', ec='black')
    ax.bar(x - width / 2, data.loc[configurations]['RL Energy battery [kWh]'], width,
           color='lime', label='RL-Battery', ec='black')
    ax.bar(x - width / 2, data.loc[configurations]['RL Energy from PV [kWh]'], width,
           bottom=data.loc[configurations]['RL Energy battery [kWh]'],
           color='yellow', label='RL-PV', ec='black')
    ax.bar(x - width / 2, data.loc[configurations]['RL Energy from grid [kWh]'], width,
           bottom=data.loc[configurations]['RL Energy battery [kWh]']+data.loc[configurations]['RL Energy from PV [kWh]'],
           color='red', label='RL-Grid', ec='black')

    # ax.bar(x + width / 2, data.loc[configurations]['RBC Energy consumption [kWh]'], width,
    #        color='cornflowerblue', label='RBC-Consumption', ec='black')
    ax.bar(x + width / 2, data.loc[configurations]['RBC Energy battery [kWh]'], width,
           color='limegreen', label='RBC-Battery', ec='black')
    ax.bar(x + width / 2, data.loc[configurations]['RBC Energy from PV [kWh]'], width,
           bottom=data.loc[configurations]['RBC Energy battery [kWh]'],
           color='khaki', label='RBC-PV', ec='black')
    ax.bar(x + width / 2, data.loc[configurations]['RBC Energy from grid [kWh]'], width,
           bottom=data.loc[configurations]['RBC Energy battery [kWh]']+data.loc[configurations]['RBC Energy from PV [kWh]'],
           color='indianred', label='RBC-Grid', ec='black')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Tank volume')
    ax.set_ylabel('Energy [kWh]')
    ax.set_title(f'Energy consumption and from battery for RB and RL at {size} Wh and best seed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', ncol=1, fontsize='small')
    ax.set_ylim(0.0, 1500.0)

    plt.savefig(directory_plot + f'Energy consumption and from battery for RB and RL at {size} Wh and best seed.png')
    plt.show()
    plt.close()


def plot_bar_tank(configurations, labels, width=0.15):

    x = np.arange(len(labels))  # the label locations
    size = data.loc[configurations[0]]['tank_volume']

    _, ax = plt.subplots()
    ax.bar(x - 3 * width / 2, data.loc[configurations]['RL Self-Sufficiency'], width,
           color='steelblue', label='RL: Self-Sufficiency', ec='black')
    ax.bar(x - width / 2, data.loc[configurations]['RL Self-Consumption'], width,
           color='skyblue', label='RL: Self-Consumption', ec='black')
    ax.bar(x + width / 2, data.loc[configurations]['RBC Self-Sufficiency'], width,
           color='forestgreen', label='RBC: Self-Sufficiency', ec='black')
    ax.bar(x + 3 * width / 2, data.loc[configurations]['RBC Self-Consumption'], width,
           color='greenyellow', label='RBC: Self-Consumption', ec='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Battery capacity')
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f'SS and SC for RB and RL at {size} m^3 and best seed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', ncol=1, fontsize='small')
    plt.savefig(directory_plot + f'SS and SC for RB and RL at {size} m^3 and best seed.png')
    plt.close()

    _, ax = plt.subplots()

    ax.bar(x - width / 2, data.loc[configurations]['RL Energy consumption [kWh]'], width,
           color='steelblue', label='RL-Consumption', ec='black')
    ax.bar(x - width / 2, data.loc[configurations]['RL Energy battery [kWh]'], width,
           color='skyblue', label='RL-Battery', ec='black')
    ax.bar(x + width / 2, data.loc[configurations]['RBC Energy consumption [kWh]'], width,
           color='forestgreen', label='RBC-Consumption', ec='black')
    ax.bar(x + width / 2, data.loc[configurations]['RBC Energy battery [kWh]'], width,
           color='greenyellow', label='RBC-Battery', ec='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Battery capacity')
    ax.set_ylabel('Energy [kWh]')
    ax.set_title(f'Energy consumption and from battery for RB and RL at {size} m^3 and best seed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', ncol=1, fontsize='small')
    ax.set_ylim(0.0, 1500.0)

    plt.savefig(directory_plot + f'Energy consumption and from battery for RB and RL at {size} m^3 and best seed.png')
    plt.show()
    plt.close()
########################################################################################################################


labels_volume = ['10 m^3', '8 m^3', '6 m^3']
labels_energy = ['2400 Wh', '4800 Wh', '7200 Wh']

# Battery 2400 Wh, Configuration
configuration = [0, 3, 6]
plot_bar_battery(configuration, labels_volume)
# Battery 4800 Wh, Configuration
configuration = [1, 4, 7]
plot_bar_battery(configuration, labels_volume)
# Battery 7200 Wh, Configuration
configuration = [2, 5, 8]
plot_bar_battery(configuration, labels_volume)


# Tank 10, Configuration
configuration = [0, 1, 2]
plot_bar_tank(configuration, labels_energy)
# Tank 8, Configuration
configuration = [3, 4, 5]
plot_bar_tank(configuration, labels_energy)
# Tank 6, Configuration
configuration = [6, 7, 8]
plot_bar_tank(configuration, labels_energy)

