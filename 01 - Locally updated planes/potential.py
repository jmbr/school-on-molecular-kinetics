import sys
import numpy as np
import matplotlib.pyplot as plt


epsilon = sys.float_info.epsilon
threshold = 1e-3


class Potential:
    '''Muller potential.'''
    def __call__(self, z):
        return -200*np.exp(-(z[0]-1)**2-10*z[1]**2)-100*np.exp(-z[0]**2-10*(z[1]-1/2)**2)-170*np.exp(-(13/2)*(z[0]+1/2)**2+11*(z[0]+1/2)*(z[1]-3/2)-(13/2)*(z[1]-3/2)**2)+15*np.exp((7/10)*(z[0]+1)**2+(3/5)*(z[0]+1)*(z[1]-1)+(7/10)*(z[1]-1)**2)

    def force(self, z):
        '''Force exerted by the Muller potential.'''
        return -np.array([-200*(-2*z[0]+2)*np.exp(-(z[0]-1)**2-10*z[1]**2)+200*z[0]*np.exp(-z[0]**2-10*(z[1]-1/2)**2)-170*(-13*z[0]-23+11*z[1])*np.exp(-(13/2)*(z[0]+1/2)**2+11*(z[0]+1/2)*(z[1]-3/2)-(13/2)*(z[1]-3/2)**2)+15*((7/5)*z[0]+4/5+(3/5)*z[1])*np.exp((7/10)*(z[0]+1)**2+(3/5)*(z[0]+1)*(z[1]-1)+(7/10)*(z[1]-1)**2),
                          4000*z[1]*np.exp(-(z[0]-1)**2-10*z[1]**2)-100*(-20*z[1]+10)*np.exp(-z[0]**2-10*(z[1]-1/2)**2)-170*(11*z[0]+25-13*z[1])*np.exp(-(13/2)*(z[0]+1/2)**2+11*(z[0]+1/2)*(z[1]-3/2)-(13/2)*(z[1]-3/2)**2)+15*((3/5)*z[0]-4/5+(7/5)*z[1])*np.exp((7/10)*(z[0]+1)**2+(3/5)*(z[0]+1)*(z[1]-1)+(7/10)*(z[1]-1)**2)])

    def plot(self, x0=-1.85, x1=1.25, y0=-0.5, y1=2.25,
             min_value=-150, max_value=20, nx=500, ny=500):
        '''Plot Muller potential.'''
        xx = np.linspace(x0, x1, nx)
        yy = np.linspace(y0, y1, ny)
        x, y = np.meshgrid(xx[:-1], yy[:-1])

        z = self.__call__(np.array([x, y]))

        zz = z.clip(min=min_value, max=max_value)
        zz[zz == min_value] = np.nan
        zz[zz == max_value] = np.nan

        plt.contourf(x, y, zz, 1000, cmap='Blues',
                     vmin=min_value, vmax=max_value)

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(label='Energy')

        plt.xlim([x0, x1])
        plt.ylim([y0, y1])
        plt.axes().set_aspect('equal')
