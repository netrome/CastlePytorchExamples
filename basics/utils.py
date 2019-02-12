import numpy as np
import random

mus = {
    0: np.array([-1, -1]),
    1: np.array([1, 1]),
}

sigmas = {
    0: np.array([[1, 0.6], [0.6, 1]]),
    1: np.array([[1, -0.6], [-0.6, 1]]),
}

def sample_a_data_point():
    y = np.random.choice([0, 1])
    x = np.random.multivariate_normal(mus[y], sigmas[y])

    return x, y


if __name__=="__main__":
    import matplotlib.pyplot as plt


    for i in range(1000):
        point = sample_a_data_point()
        c = 'r*' if point[1] == 0 else 'b*'
        x, y = point[0]
        plt.plot([x], [y], c)

    plt.show()


