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

price = pd.read_csv(directory+'electricity_price_schedule.csv', sep=';', decimal='.', index_col=None)
price = price.to_numpy().squeeze()

high = (price == 0.3)
medium = (price == 0.165)
low = (price == 0.03)

assex = data_rl.index.to_numpy()
asse_price = assex
fare = data_rl['Fare'].to_numpy(dtype=bool)


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 6), tight_layout=True)

fig.suptitle('Storage tank operation')

ax1.fill_between(assex, data_rl['STORAGETANK:Chilled Water Thermal Storage Use Side Heat Transfer Energy [J](TimeStep)'].mul(1/3600000), alpha=1, color='gold', ec='black', linewidth=1)
ax1.fill_between(assex, data_rl['STORAGETANK:Chilled Water Thermal Storage Source Side Heat Transfer Energy [J](TimeStep)'].mul(1/3600000), alpha=1, color='crimson', ec='black', linewidth=1)
ax1.fill_between(assex, price*100, -price*100, where=low, alpha=0.05, color='g')  #, interpolate=True, step='mid'
ax1.fill_between(assex, price*50, -price*50, where=medium, alpha=0.05, color='y')
ax1.fill_between(assex, price*10, -price*10, where=high, alpha=0.05, color='r')  #, interpolate=True, step='mid'
ax1.set_ylabel('Thermal power [kW]')
ax1.set_ylim([-10, 10])
ax1.grid(axis='y', alpha=0.15, linewidth=0.5, color='black')

ax2.plot(assex, data_rl['Tank SOC'])
ax2.fill_between(assex, price*999999, -price*999999, where=(price <= 0.03), alpha=0.05, color='g')  #, step='post', interpolate='True')
ax2.fill_between(assex, price*999999, -price*999999, where=(price == 0.165), alpha=0.05, color='y')  #, step='pre', interpolate='True')
ax2.fill_between(assex, price*999999, -price*999999, where=(price >= 0.3), alpha=0.05, color='r')  #, step='pre', interpolate='True')
ax2.set_xlabel('Time')
ax2.set_ylabel('State of Charge')
ax2.set_ylim([-0.05, 1.1])
ax2.set_xticks(assex[fare])
ax2.set_xticklabels(assex[fare], rotation=90)
ax2.grid(axis='y', alpha=0.15, linewidth=0.5, color='black')

for xc in assex[fare]:
    ax1.axvline(x=xc, linestyle=(0, (5, 5)), color='black', linewidth=0.75)
    ax2.axvline(x=xc, linestyle=(0, (5, 5)), color='black', linewidth=0.75)
plt.xlim([' 07/03  01:00:00', ' 07/10  01:00:00'])
plt.show()
