class Battery:
    def __init__(self, max_power, max_capacity, rte, soc_min=0.04, soc_max=0.96, eta_dc_dc=0.95):
        self.max_power = max_power
        self.max_capacity = max_capacity  # Max capacity = 2h storage = 2 x max power Wh
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
