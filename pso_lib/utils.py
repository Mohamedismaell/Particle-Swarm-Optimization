import numpy as np


def calculate_distance(d, p):
    return np.sqrt((d['x'] - p['x'])**2 + (d['y'] - p['y'])**2)
