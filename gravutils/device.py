#IMPORTS
from host import State, HostState

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.tools as tools
import pycuda.autoinit
import numpy as np

from scikits.cuda.cublas import *
from pycuda.compiler import SourceModule

"""
The back-end for the gravity simulator
"""

#GLOBALS
THETA = np.float32(1.0/(2-2**(1/3.0)))
G_norm = np.float32(6.667*10**(-11))


class DeviceState(State):
    """Represents the physical state of the system on the device"""

    #For the acceleration kernel
    get_accelerations = None
    block_params = (400,1,1)
    grid_params = (16,1)

    a_vec_temp = None

    def __init__(self, mass, pos, vel):
        super(State, self).__init__()

        self.mass = gpuarray.to_gpu(mass)
        self.pos = gpuarray.to_gpu(pos)
        self.vel = gpuarray.to_gpu(vel)
        self.a_vec_temp = np.zeros_like(pos)

        self.N = np.int32(mass.size)

        #INITIALIZE LINEAR ALGEBRA
        self.h = cublasCreate()

        #Import the acceleration kernel
        with open("/home/aidan/Programming/cuda/py-gravity/gravutils/accel_kernel.cu", "r") as f:
            self.module = SourceModule(f.read())
        self.get_accelerations = self.module.get_function("get_accelerations")

    def __del__(self):#
        #CLEANUP LINEAR ALGEBRA
        cublasDestroy(self.h)

        #Clean up variables
        del self.mass
        del self.pos
        del self.vel

    def sync_with_host(self, hstate):
        self.mass = hstate.mass
        self.pos = gpuarray.to_gpu(hstate.pos)
        self.vel = gpuarray.to_gpu(hstate.vel)

    @classmethod
    def from_host_state(cls, hstate):
        return cls(hstate.mass, hstate.pos, hstate.vel)

    def step(self, dt, G=G_norm):
        """
        Steps the system forward by dt using a 4th order forrest-ruth
        symplectic integrator.
        """

        #body.loc = body.loc +          THETA  * dt/2 * body.vel
        self.saxpy(THETA*dt/2 , self.vel, self.pos)

        #body.vel = body.vel +          THETA  * dt   * self.acceleration(body.loc, body, states)
        a_vec = self.acceleration(self.pos)
        self.saxpy(THETA*dt*G , a_vec, self.vel)

        #body.loc = body.loc + (1   -   THETA) * dt/2 * body.vel
        self.saxpy((1-THETA)*dt/2 , self.vel, self.pos)

        #body.vel = body.vel + (1   - 2*THETA) * dt   * self.acceleration(body.loc, body, states)
        a_vec = self.acceleration(self.pos)
        self.saxpy((1-2*THETA)*dt*G , a_vec, self.vel)

        #body.loc = body.loc + (1   -   THETA) * dt/2 * body.vel
        self.saxpy((1-THETA)*dt/2 , self.vel, self.pos)

        #body.vel = body.vel +          THETA  * dt   * self.acceleration(body.loc, body, states)
        a_vec = self.acceleration(self.pos)
        self.saxpy(THETA*dt*G , a_vec, self.vel)

        #body.loc = body.loc +          THETA  * dt/2 * body.vel
        self.saxpy(THETA*dt/2 , self.vel, self.pos)

    def acceleration(self, pos):
        """Gets the acceleration for a given position matrix, and stores it in a_vec"""
        self.get_accelerations(
            drv.Out(self.a_vec_temp), drv.In(pos.get()), drv.In(self.mass.get()), self.N,
            block=self.block_params, grid=self.grid_params)
        return gpuarray.to_gpu(self.a_vec_temp)

    def saxpy(self,a,x,y,xs=1,ys=1):
        cublasSaxpy(self.h, y.size, a, x.gpudata, xs, y.gpudata, ys)
