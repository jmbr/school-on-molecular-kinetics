import numpy as np


class Timestepper:
    def __init__(self, potential, beta, dt, initial_value):
        self.potential = potential
        self.beta = beta
        self.dt = dt
        self.x = initial_value
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        Fdt = self.potential.force(self.x) * self.dt
        R = np.sqrt(2.0 / self.beta * self.dt) * np.random.randn()

        self.x += Fdt + R
        self.n += 1

        return self


class FirstPassageTime:
    def __init__(self, potential, beta, dt, z, a, b):
        self.potential = potential
        self.beta = beta
        self.dt = dt
        self.z = z
        self.a = a
        self.b = b

    def run_trajectory_fragment(self):
        steps = Timestepper(self.potential, self.beta, self.dt,
                            self.z)

        for step in steps:
            x = step.x
            if x <= self.a or x >= self.b:
                return step.n * self.dt

    def mean(self, num_trajectories):
        fpt = np.zeros(num_trajectories)

        for n in range(num_trajectories):
            fpt[n] = self.run_trajectory_fragment()

        return fpt.mean(), fpt.std()
