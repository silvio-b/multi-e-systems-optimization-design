import pandas as pd
import matplotlib.pyplot as plt

directory = 'C:\\Users\\agall\\OneDrive\\Desktop\\Results_PV2000\\'
directory_plot = 'C:\\Users\\agall\\OneDrive\\Desktop\\Plots\\'
# Line plot for storage tank and battery over time
# Area plot for  charge/Discharge power for tank and battery

day = range(23+24*5, 2160, 8)
fare = range(23+24*5, 2160, 24*7)

data_rl = pd.read_csv(directory+'test_09/configuration_008/episode_25.csv', sep=';', decimal=',', index_col=0)
data_rb = pd.read_csv(directory+'test_09/configuration_008/baseline.csv', sep=';', decimal=',', index_col=0)

assex = data_rl.index.to_numpy()
price = data_rl['Price'].to_numpy()
fare = data_rl['Fare'].to_numpy(dtype=bool)

low = (price == 0.03)
medium = (price == 0.165)
high = (price == 0.3)


def add_true(schedule):
    for index, tariff in list(enumerate(schedule)):
        if tariff and index > 0:
            schedule[index-1] = True
    return schedule


low = add_true(low)
medium = add_true(medium)
high = add_true(high)


def plot_operation(data, tank_volume, battery_capacity, control_strategy):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 6), tight_layout=True)

    fig.suptitle(f'Tank operation by {control_strategy}')

    ax1.fill_between(assex, data['STORAGETANK:Chilled Water Thermal Storage Use Side Heat Transfer Energy [J](TimeStep)'].mul(1/3600000),
                     label='Discharge power', alpha=1, color='gold', ec='black', linewidth=1)
    ax1.fill_between(assex, data['STORAGETANK:Chilled Water Thermal Storage Source Side Heat Transfer Energy [J](TimeStep)'].mul(1/3600000),
                     label='Charge power', alpha=1, color='crimson', ec='black', linewidth=1)
    ax1.fill_between(assex, price*10000, -price*10000, where=low, alpha=0.05, color='g')  #, step='mid'
    ax1.fill_between(assex, price*5000, -price*5000, where=medium, alpha=0.05, color='y')
    ax1.fill_between(assex, price*1000, -price*1000, where=high, alpha=0.05, color='r')  #, interpolate=True, step='mid'
    ax1.set_ylabel('Thermal power [kW]', fontsize='large')
    ax1.set_ylim([-10, 10])
    ax1.grid(axis='y', alpha=0.15, linewidth=0.5, color='black')
    ax1.legend(bbox_to_anchor=(1.08, 1.0), loc='upper center', ncol=1)

    ax2.plot(assex, data['Tank SOC'])
    ax2.fill_between(assex, price*999999, -price*999999, where=low, alpha=0.05, color='g', label='Low price')  #, step='post', interpolate='True')
    ax2.fill_between(assex, price*999999, -price*999999, where=medium, alpha=0.05, color='y', label='Medium price')  #, step='pre', interpolate='True')
    ax2.fill_between(assex, price*999999, -price*999999, where=high, alpha=0.05, color='r', label='High price')  #, step='pre', interpolate='True')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('State of Charge', fontsize='large')
    ax2.set_ylim([-0.05, 1.1])
    ax2.set_xticks(assex[fare])
    ax2.set_xticklabels(assex[fare], rotation=90, fontsize=6)
    ax2.grid(axis='y', alpha=0.15, linewidth=0.5, color='black')
    ax2.legend(bbox_to_anchor=(1.07, 1.0), loc='upper center', ncol=1)

    for xc in assex[fare]:
        ax1.axvline(x=xc, linestyle=(0, (5, 5)), color='black', linewidth=0.75)
        ax2.axvline(x=xc, linestyle=(0, (5, 5)), color='black', linewidth=0.75)
    plt.xlim([' 07/03  01:00:00', ' 07/10  01:00:00'])
    plt.savefig(f'{directory_plot}Tank operation by {control_strategy} with {tank_volume} m3 and {battery_capacity} Wh.png')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 6), tight_layout=True)

    fig.suptitle(f'Battery operation by {control_strategy}')

    ax1.fill_between(assex, data['battery energy to building [J]'].mul(-1/3600000),
                     label='Discharge power', alpha=1, color='gold', ec='black', linewidth=1)
    ax1.fill_between(assex, data['PV energy to battery [J]'].mul(1/3600000),
                     label='Charge power', alpha=1, color='crimson', ec='black', linewidth=1)
    ax1.fill_between(assex, price*10000, -price*10000, where=low, alpha=0.05, color='g')  #, step='mid'
    ax1.fill_between(assex, price*5000, -price*5000, where=medium, alpha=0.05, color='y')
    ax1.fill_between(assex, price*1000, -price*1000, where=high, alpha=0.05, color='r')  #, interpolate=True, step='mid'
    ax1.set_ylabel('Electric power [kW]', fontsize='large')
    ax1.set_ylim([-4, 4])
    ax1.grid(axis='y', alpha=0.15, linewidth=0.5, color='black')
    ax1.legend(bbox_to_anchor=(1.08, 1.0), loc='upper center', ncol=1)

    ax2.plot(assex, data['Battery soc'])
    ax2.fill_between(assex, price*999999, -price*999999, where=low, alpha=0.05, color='g', label='Low price')  #, step='post', interpolate='True')
    ax2.fill_between(assex, price*999999, -price*999999, where=medium, alpha=0.05, color='y', label='Medium price')  #, step='pre', interpolate='True')
    ax2.fill_between(assex, price*999999, -price*999999, where=high, alpha=0.05, color='r', label='High price')  #, step='pre', interpolate='True')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('State of Charge', fontsize='large')
    ax2.set_ylim([-0.05, 1.1])
    ax2.set_xticks(assex[fare])
    ax2.set_xticklabels(assex[fare], rotation=90, fontsize=6)
    ax2.grid(axis='y', alpha=0.15, linewidth=0.5, color='black')
    ax2.legend(bbox_to_anchor=(1.07, 1.0), loc='upper center', ncol=1)

    for xc in assex[fare]:
        ax1.axvline(x=xc, linestyle=(0, (5, 5)), color='black', linewidth=0.75)
        ax2.axvline(x=xc, linestyle=(0, (5, 5)), color='black', linewidth=0.75)
    plt.xlim([' 07/03  01:00:00', ' 07/10  01:00:00'])
    plt.savefig(f'{directory_plot}Battery operation by {control_strategy} with {tank_volume} m3 and {battery_capacity} Wh.png')
    plt.close()


best_seed = pd.read_csv(directory+'best_seeds.csv', sep=';', decimal=',', index_col=None)


for ii in range(0, len(best_seed)):
    configuration = best_seed.loc[ii]['directory']
    best_episode = best_seed.loc[ii]['best_episode']
    volume = best_seed.loc[ii]['tank_volume']
    capacity = best_seed.loc[ii]['battery_size']

    data_rl = pd.read_csv(directory + configuration + 'episode_' + str(best_episode) + '.csv', sep=';', decimal=',', index_col=0)
    data_rb = pd.read_csv(directory + configuration + 'baseline.csv', sep=';', decimal=',', index_col=0)
    plot_operation(data_rl, volume, capacity, 'RL')
    plot_operation(data_rb, volume, capacity, 'RB')
