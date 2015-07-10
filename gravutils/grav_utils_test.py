import numpy as np
from host import HostState
from device import DeviceState
import threading
from pycuda import driver

N = np.int32(100)

def step_test():
    self.ctx  = driver.Device(0).make_context()
    self.device = self.ctx.get_device()
    mass = np.random.randn(3*N).astype(np.float32)
    pos = np.random.randn(3*N).reshape(N,3).astype(np.float32)
    vel = np.random.randn(3*N).reshape(N,3).astype(np.float32)

    hs = HostState(mass, pos, vel)
    ds = DeviceState(mass, pos, vel)

    ds.step(np.float32(0.1))
    self.ctx.pop()
    del self.ctx

class ThreadTester(threading.Thread):
    def __init__(self, gpuid=0):
        threading.Thread.__init__(self)

    def run(self):
        self.ctx  = driver.Device(gpuid).make_context()
        self.device = self.ctx.get_device()
        #step_test()
        self.ctx.pop()
        del self.ctx

def thread_test():
    a  = ThreadTester()
    a.start()
    a.join()

print("Running step test")
step_test()
print("Completed step test")

print("Running thread test")
thread_test()
print("Completed thread test")
