import os
import pandas as pd
from EnergyModels.pv_utils import sin, cos


class PV:
    # STC sta per standard test condition, tilt angle è l inclinazione del pannello rispetto al piano orizzontale
    # l azimuth è la distanza angolare dall'asse Sud, positivo nel verso West,input angles are in degree
    def __init__(self, nominal_power, tilt_angle, azimuth, latitude=45.2, longitude=7.65, rho=0, eta_conv=0.95):
        super(PV, self).__init__()
        self.stc_power = nominal_power
        self.eta_conv = eta_conv

        self.tilt = tilt_angle
        self.azimuth = azimuth
        self.latitude = latitude
        self.longitude = longitude

        self.sky_view_factor = (1 + cos(self.tilt)) / 2  # Portion of sky seen by pv

        self.ground_view_factor = (1 - cos(self.tilt)) / 2  # Portion of ground seen by pv
        self.ground_reflectivity = rho  # rho controlla nome

        self.stc_efficiency = self.efficiency_compute()
        self.surface = self.stc_power / self.stc_efficiency / 1000

        directory = os.path.dirname(os.path.realpath(__file__))

        self.angles = pd.read_csv(directory + '\\sol_pos.csv', decimal=',', sep=';')

    def solar_angles_calculation(self, day, time):
        # Day is the number of the day of the year, time input is in h but need in minutes

        if time == 24:
            time = 0
        time = round(time * 60)

        angle = self.angles[self.angles['Day'] == day]
        angle = angle.loc[self.angles['Time'] == time]

        zenith = angle['zenith'].item()
        azimuth_sun = angle['azimuth'].item()

        incidence_angle_cosine = max(0, cos(zenith) * cos(self.tilt) + sin(zenith) * sin(self.tilt)
                                     * cos(azimuth_sun - self.azimuth))

        return incidence_angle_cosine, zenith

    def efficiency_compute(self, am=1.5, irradiance=1000, ambient_temperature=25,
                           p=23.62, q=-0.2983, m=0.1912, r=-0.09307, s=-0.9795, u=0.9865, h=0.028):
        # Irradiance in W/m2, temperature in degree
        g_stc = 1000
        theta_stc = 25
        am_stc = 1.5
        cell_temperature = ambient_temperature + h * irradiance  # Ross Model see reference
        efficiency = p * (q * irradiance / g_stc + (irradiance / g_stc)**m) \
                     * (1 + r * cell_temperature / theta_stc + s * am / am_stc + (am / am_stc)**u)
        return efficiency/100

    def electricity_prediction(self, direct_radiation, diffuse_radiation, t_out, day, time):

        incidence_angle_cosine, zenith = self.solar_angles_calculation(day, time)
        if zenith > 90:
            zenith = 90

        am = 1 / (cos(zenith) + 0.50572 * (96.07995 - zenith)**(-1.6364))

        # 3 component
        direct = direct_radiation * incidence_angle_cosine
        diffuse = diffuse_radiation * self.sky_view_factor
        ground_reflection = (direct_radiation + diffuse_radiation) * self.ground_view_factor * self.ground_reflectivity

        radiation = direct + diffuse + ground_reflection

        efficiency = self.efficiency_compute(am, radiation, t_out)
        power = radiation * efficiency * self.surface * self.eta_conv
        return power, efficiency
