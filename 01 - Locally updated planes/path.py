import sys
import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


epsilon = sys.float_info.epsilon
default_threshold = 1e-3


tmax = 10
timesteps = 1000


class Path:
    '''Reaction path.'''
    def __init__(self, reactant, product, num_nodes, potential,
                 threshold=default_threshold):
        assert num_nodes >= 2
        assert not math.isclose(np.linalg.norm(reactant - product),
                                0.0, rel_tol=epsilon)
        assert reactant.shape == product.shape

        self.reactant = reactant
        self.product = product
        self.num_nodes = num_nodes
        self.potential = potential
        self.threshold = threshold
        self.energy = np.inf
        self.reset()

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        prev_energy = self.energy
        self.update()
        curr_energy = self.energy
        diff_energy = curr_energy - prev_energy
        print('Current energy: {:g}'.format(curr_energy))
        if np.abs(diff_energy) / np.abs(curr_energy) < self.threshold:
            raise StopIteration
        else:
            return self

    def reset(self):
        '''Initialize path by linear interpolation.'''
        def initial_path(t):
            return (1 - t) * self.reactant + t * self.product

        self.points = np.zeros((self.num_nodes,
                                self.reactant.shape[0]))

        self.energy = 0.0

        for k, t in enumerate(np.linspace(0, 1, self.num_nodes)):
            self.points[k, :] = initial_path(t)
            self.energy += self.potential(self.points[k, :])

        return self

    def update(self):
        '''Construct a new path by minimization (steepest descent) along the
        orthogonal complement to the curve.

        '''
        new_points = self.points.copy()

        self.energy = 0.0

        for k in range(self.num_nodes):
            if k == 0 or k == self.num_nodes - 1:
                continue

            z = self.points[k, :]
            v = ((self.points[k+1, :] - self.points[k-1, :]) /
                 np.linalg.norm(self.points[k+1, :] - self.points[k-1, :]))

            def vector_field(z, t):
                f = self.potential.force(z)
                return f - v * np.dot(v, f)

            t = np.linspace(0, tmax, timesteps)
            sol = scipy.integrate.odeint(vector_field, z, t)
            new_points[k, :] = sol[-1, :]

            self.energy += self.potential(new_points[k, :])

        self.points = new_points

    def plot(self):
        '''Plot path.'''
        self.potential.plot()
        plt.plot(self.points[:, 0], self.points[:, 1], 'o-', color='black')
