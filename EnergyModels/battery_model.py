class Battery:
    def __init__(self, max_capacity, rte, soc_min=0.10, soc_max=0.90, eta_dc_dc=0.95):
        # Max capacity in Wh
        self.max_power_charging = 0.5 * max_capacity  # In W
        self.max_power_discharging = max_capacity  # In W
        self.max_capacity = max_capacity
        self.rte = rte
        self.eta_dc_dc = eta_dc_dc
        self.soc_min = soc_min
        self.soc_max = soc_max

        self.soc = (self.soc_max - self.soc_min) / 2

    def charge(self, energy):  # energy deve essere in J, è riferita a quella prima del converter
        energy = abs(energy/3600)  # Passa in Wh
        self.soc += energy / self.max_capacity * self.rte * self.eta_dc_dc

    def discharge(self, energy):  # energy deve essere in J
        energy = abs(energy/3600)  # Passo in Wh
        self.soc -= energy / self.max_capacity / self.eta_dc_dc
