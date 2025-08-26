

import numpy as np
import random


def square_wave_mechanism( angle, epsilon):

    angle_max = 10
    t = angle / angle_max
    b = (epsilon * (np.e ** (epsilon)) - np.e ** epsilon + 1) / (
            2 * np.e ** (epsilon) * (np.e ** epsilon - 1 - epsilon))
    x = random.uniform(0, 1)
    if x < (2 * b * np.e ** epsilon) / (2 * b * np.e ** epsilon + 1):
        perturbed_t = random.uniform(- b + t, b + t)
    else:
        x_1 = random.uniform(0, 1)
        if x_1 <= t:
            perturbed_t = random.uniform(- b, - b + t)
        else:
            perturbed_t = random.uniform(b + t, b + 1)
    perturbed_t = (perturbed_t + b) / (2 * b + 1)
    perturbed_t = perturbed_t * angle_max

    return perturbed_t
