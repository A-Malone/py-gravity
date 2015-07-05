import numpy as np
from host import HostState
import threading

N = np.int32(100)

def step_test():
    from device import DeviceState
    mass = np.random.randn(3*N).astype(np.float32)
    pos = np.random.randn(3*N).reshape(N,3).astype(np.float32)
    vel = np.random.randn(3*N).reshape(N,3).astype(np.float32)

    print("Creating objects")
    hs = HostState(mass, pos, vel)
    ds = DeviceState(mass, pos, vel)
    print("Created objects")

    print("Stepping")
    ds.step(np.float32(0.1))
    print("Finished")

class ThreadTester(threading.Thread):
    def run(self):
        step_test()

def thread_test():
    a  = ThreadTester()
    a.start()

print("Running step test")
step_test()
print("Completed step test")

print("Running thread test")
thread_test()
print("Completed thread test")
