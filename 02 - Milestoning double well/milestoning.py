import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

import matplotlib.pyplot as plt


class Potential:
    '''Double well with prescribed barrier height and asymmetric depth.

    '''
    def __init__(self, height, depth):
        self.height = height
        self.depth = depth

    def __call__(self, x):
        h, k = self.height, self.depth
        return ((x + 1)**2 *
                (x**4 + (0.75 * k - 2) * x**3 +
                 (h - k + 1) * x**2 - 2 * h * x + h))

    def force(self, x):
        h, k = self.height, self.depth
        return -(3.75 * k * x**4 -
                 3.75 * k * x**2 +
                 6 * x**5 +
                 (4*h + 2.0*k - 8) * x**3 +
                 (2 - 4 * h - 2 * k) * x)

    def plot(self):
        xmin, xmax, n = -2, 2, 1000
        x = np.linspace(xmin, xmax, n)
        y = self.__call__(x)
        plt.plot(x, y)
        plt.xlim([xmin, xmax])
        plt.ylim([-self.depth - 0.25, self.height + 0.25])
        plt.xlabel('$x$')
        plt.ylabel('Potential energy')


default_n = 250


class OccupationDensity():
    '''Compute the occupation density.

    This function computes the occupation density corresponding to the
    given potential energy for an initial distribution with total
    "mass" equal to q placed at the point a < x0 < b with absorbing
    boundary conditions at the endpoints of the interval (a, b).
    '''

    def __init__(self, potential, beta, a, b, x0, n=default_n):
        self.potential = potential
        self.beta = beta
        self.a = a
        self.b = b
        self.x0 = x0
        self.n = n
        self.func = None

    def __call__(self, x):
        '''Evaluates the density function at the given points.

        Since it is computationally expensive to evaluate the true
        density function, we tabulate it on $n$ values.

        '''
        if self.func is None:
            f = self.__make_density()
            z = np.linspace(self.a, self.b, self.n)
            self.func = interpolate.interp1d(z, f(z),
                                             bounds_error=False,
                                             fill_value=0.0)

        return self.func(x)

    def __make_density(self):
        '''Compute density function.

        '''
        heaviside = np.vectorize(self.__heaviside)

        U = self.potential

        quad = integrate.quad

        def auxint(w):
            return np.exp(self.beta * U(w))

        aux1 = quad(auxint, self.a, self.x0, full_output=1)
        aux2 = quad(auxint, self.x0, self.b, full_output=1)
        aux = aux1[0] + aux2[0]

        def aux3(x):
            if self.a < self.x0 < x:
                singularities = [self.x0]
            else:
                singularities = None
            val = quad(lambda w: (aux2[0] / aux - heaviside(w - self.x0))
                       * np.exp(self.beta * U(w)),
                       self.a, x, points=singularities, full_output=1)
            return self.beta * val[0]

        return np.vectorize(lambda z: np.exp(-self.beta * U(z)) * aux3(z))

    def plot(self):
        n = 1000
        x = np.linspace(self.a, self.b, n)
        y = self.__call__(x)
        plt.plot(x, y)
        plt.xlim([self.a, self.b])
        plt.ylim([np.min(y) - 0.25, np.max(y) + 0.25])
        plt.xlabel('$x$')
        plt.ylabel('Occupation density')

    def __heaviside(self, x):
        '''Heaviside step function.'''
        if x < 0.0:
            return 0.0
        elif x == 0.0:
            return 0.5
        else:
            return 1.0
