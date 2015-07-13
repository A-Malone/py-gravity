from device import DeviceState
from host import HostState, Body
import numpy as np
import json
import threading, time
import pygame

class GravSim(object):
    """ The physics simulator object """

    #----PHYSICS SCALING
    #   Time scale      =>  Earth min
    #   Length scale    =>  AU / 1000
    #   Mass scale      =>  Earth Mass
    MASS_SACLE = 5.972*10**24       #kg/MU
    LENGTH_SCALE = 149597870.700    #m/LU
    TIME_SCALE = 60.0               #s/Tu

    #----CONSTANTS
    AU = 149597870700.0    #m/AU

    DT = 360  #1 hour timestep

    #----CONSTANTS
    AU = 149597870700.0    #m/AU

    hostState = None
    deviceState = None

    def __init__(self, timestep=None):

        #Init the super class
        super(GravSim, self).__init__()

        #Create a lock object
        self.phys_lock = threading.Lock()
        self.stopped = False

        #Timstep
        self.dt = timestep or self.DT

        self.G = np.float32(6.67*10**(-11) * self.MASS_SACLE / self.LENGTH_SCALE **3 * (self.TIME_SCALE)**2) #(LU)^3/(MU)/(TU)^2

        #----SET UP SIMLUATOR

        #SOLAR SYSTEM DATA
        #   posvel[i][2:0] => positions x,y,z in AU
        #   posvel[i][5:3] => velocity  x,y,z in AU/day
        with open("./data.json", 'r') as fp:
            json_data = json.load(fp)

        #Load data from the file
        pos = np.asarray(json_data["posvel"])[:,:3].astype(np.float32) * self.AU / self.LENGTH_SCALE
        vel = np.asarray(json_data["posvel"])[:,3:].astype(np.float32) * self.AU / self.LENGTH_SCALE / 24.0 / 60.0 / (60 / self.TIME_SCALE)
        mass = np.asarray(json_data["mass"]).astype(np.float32)
        print(pos)
        print(vel)

        #Setup the host state object
        self.hostState = HostState(mass,pos,vel)

        #Load Body information
        for index,body in enumerate(json_data["bodies"]):
            self.hostState.add_body( Body(index, body) )

        #Setup and start the clock
        self.clock = pygame.time.Clock()
        self.time = 0


    def run(self):
        #Setup the device state wholly within the simulator thread
        self.deviceState = DeviceState.from_host_state(self.hostState)

        print("Starting physics simulator")
        while(not self.stopped):

            #Tell the device to perform the step
            self.deviceState.step(self.dt, self.G)

            #Get the host lock and copy over the new information
            with self.phys_lock:
                self.hostState.sync_with_device(self.deviceState)

            self.time += self.dt
            self.clock.tick(50)
        print("Stopping physics simulator")

    def stop(self):
        self.stopped = True
