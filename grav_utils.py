#!/usr/bin/env python

"""
The back-end for the gravity simulator
"""

#IMPORTS
import pycuda.gpuarray as gpuarray
from pycuda.reduction import ReductionKernel
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

#import scikits.cuda.linalg as linalg
#import scikits.cuda.misc as cumisc
from scikits.cuda.cublas import *

from pycuda.compiler import SourceModule

#Import the acceleration kernel
with open("./accel_kernel.cu", "r") as f:
    module = SourceModule(f.read())

#INITIALIZE LINEAR ALGEBRA
h = cublasCreate()

#GLOBALS
THETA = np.float32(1.0/(2-2**(1/3.0)))
N = np.int32(100)
G = np.float32(6.667*10**(-11))


class Body(object):
    """
    Represent one body on the front-end
        Stores no positional data on it's own, instead stores a reference ID
    """
    def __init__(self, obj_id, name):
        self.obj_id = obj_id
        self.name = name
        self.radius = radius

    def __repr__(self):
        return 'ID:{ID} radius:{r}'.format(ID=self.obj_id, r=self.radius)


class State(object):
    """Used to represent the physical state of the system"""
    mass = None
    pos = None
    vel = None

    def __init__(self, mass, pos, vel):
        super(State, self).__init__()

class HostState(State):
    """Represents the physical state of the system on the host"""
    def __init__(self, mass, pos, vel):
        super(State, self).__init__()
        self.mass = mass
        self.pos = pos
        self.vel = vel

    def sync_with_device(self, dstate):
        self.mass = dstate.mass.get()
        self.pos = dstate.pos.get()
        self.vel = dstate.vel.get()

class DeviceState(State):
    """Represents the physical state of the system on the device"""

    #ACCELERATION
    get_accelerations = module.get_function("get_accelerations")
    block_params = (400,1,1)
    grid_params = (1,1)

    a_vec_temp = None


    def __init__(self, mass, pos, vel):
        super(State, self).__init__()
        self.mass = gpuarray.to_gpu(mass)
        self.pos = gpuarray.to_gpu(pos)
        self.vel = gpuarray.to_gpu(vel)
        self.a_vec_temp = np.zeros_like(pos)

    def sync_with_host(self, hstate):
        self.mass = hstate.mass
        self.pos = gpuarray.to_gpu(hstate.pos)
        self.vel = gpuarray.to_gpu(hstate.vel)

    def step(self, dt):
        """
        Steps the system forward by dt using a 4th order forrest-ruth
        symplectic integrator.
        """

        #body.loc = body.loc +          THETA  * dt/2 * body.vel
        saxpy(THETA*dt/2 , self.vel, self.pos)

        #body.vel = body.vel +          THETA  * dt   * self.acceleration(body.loc, body, states)
        a_vec = self.acceleration(self.pos)
        saxpy(THETA*dt*G , a_vec, self.vel)

        #body.loc = body.loc + (1   -   THETA) * dt/2 * body.vel
        saxpy((1-THETA)*dt/2 , self.vel, self.pos)

        #body.vel = body.vel + (1   - 2*THETA) * dt   * self.acceleration(body.loc, body, states)
        a_vec = self.acceleration(self.pos)
        saxpy((1-2*THETA)*dt*G , a_vec, self.vel)

        #body.loc = body.loc + (1   -   THETA) * dt/2 * body.vel
        saxpy((1-THETA)*dt/2 , self.vel, self.pos)

        #body.vel = body.vel +          THETA  * dt   * self.acceleration(body.loc, body, states)
        a_vec = self.acceleration(self.pos)
        saxpy(THETA*dt*G , a_vec, self.vel)

        #body.loc = body.loc +          THETA  * dt/2 * body.vel
        saxpy(THETA*dt/2 , self.vel, self.pos)

    def acceleration(self, pos):
        """Gets the acceleration for a given position matrix, and stores it in a_vec"""
        self.get_accelerations(
            drv.Out(self.a_vec_temp), drv.In(pos.get()), drv.In(self.mass.get()), N,
            block=self.block_params, grid=self.grid_params)
        return gpuarray.to_gpu(self.a_vec_temp)

def saxpy(a,x,y,xs=1,ys=1):
    cublasSaxpy(h, y.size, a, x.gpudata, xs, y.gpudata, ys)


def main():
    mass = np.random.randn(3*N).astype(np.float32)
    pos = np.random.randn(3*N).reshape(N,3).astype(np.float32)
    vel = np.random.randn(3*N).reshape(N,3).astype(np.float32)

    hs = HostState(mass, pos, vel)
    ds = DeviceState(mass, pos, vel)

    ds.step(np.float32(0.1))

def test_cuda():
    mass = np.random.randn(3*N).astype(np.float32)
    pos = np.random.randn(3*N).reshape(N,3).astype(np.float32)
    vel = np.random.randn(3*N).reshape(N,3).astype(np.float32)

    hs = HostState(mass, pos, vel)
    ds = DeviceState(mass, pos, vel)

    a = np.zeros_like(pos)
    get_accelerations = module.get_function("get_accelerations")
    get_accelerations(
        drv.Out(a), drv.In(ds.pos.get()), drv.In(mass), N,
        block=(400,1,1), grid=(1,1))


    cublasSaxpy(h, ds.pos.size, G, gpuarray.to_gpu(a).gpudata, 1, ds.pos.gpudata, 1)

    print(ds.pos.get())

main()
cublasDestroy(h)
