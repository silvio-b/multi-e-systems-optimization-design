import numpy as np


def sin(x, unit='degree'):
    if unit == 'degree':
        x = x * np.pi / 180
    sine = np.sin(x)
    return sine


def cos(x, unit='degree'):
    if unit == 'degree':
        x = x * np.pi / 180
    cosine = np.cos(x)
    return cosine


def noon_time(n):  # Only depends on the day
    b = (n - 1) * 360 / 365
    e = 229.2 * (0.000075 + 0.001868 * cos(b) - 0.032077 * sin(b) - 0.014615 * cos(2*b) - 0.04089 * sin(2*b))
    noon = 12 + (7.65 - 7.5)/15 - e/60 + 1

    return noon, e


def declination_compute(n):
    b = (n - 1) * 360 / 365
    declination = 0.006918 - 0.399912 * cos(b) + 0.070257 * sin(b) - 0.006758 * cos(2*b) + 0.000907 * sin(2*b) \
                  - 0.002679 * cos(3*b) + 0.00148 * sin(3*b)
    return declination * 180 / np.pi
