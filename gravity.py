import sys
import math
import threading, time

#INTERFACE
import pygame
from collections import defaultdict

#BACKEND
import numpy as np
import json
from gravutils.host import HostState, Body


#----WINDOW PARAMETERS
WIDTH, HEIGHT = 900, 600
WIDTHD2, HEIGHTD2 = WIDTH/2., HEIGHT/2.

#----DRAWING SCALE
DRAW_SCALE = 0.1

#Constant for Forest-Ruth
THETA = 1/(2 - 2**(1.0/3))

class BodyState(object):
    def __init__(self, loc, vel, mass, rad, obj_id):
        self.obj_id = obj_id
        self.loc = loc
        self.vel = vel
        self.mass = mass
        self.radius = rad

class GravSim(threading.Thread):
    """ The physics simulator object """

    #----PHYSICS SCALING
    #   Time scale      =>  Earth min
    #   Length scale    =>  AU / 1000
    #   Mass scale      =>  Earth Mass
    MASS_SACLE = 5.972*10**24       #kg/MU
    LENGTH_SCALE = 149597870.700    #m/LU
    TIME_SCALE = 60.0               #s/Tu

    DT = 60  #1 hour timestep

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

        self.G = 6.67*10**(-11) * self.MASS_SACLE / self.LENGTH_SCALE **3 * (self.TIME_SCALE)**2 #(LU)^3/(MU)/(TU)^2

        #----SET UP SIMLUATOR

        #SOLAR SYSTEM DATA
        #   posvel[i][2:0] => positions x,y,z in AU
        #   posvel[i][5:3] => velocity  x,y,z in AU/day
        with open("./data.json", 'r') as fp:
            json_data = json.load(fp)

        #Load data from the file
        pos = np.asarray(json_data["posvel"])[:,:3].astype(np.float32)
        vel = np.asarray(json_data["posvel"])[:,3:].astype(np.float32)
        mass = np.asarray(json_data["mass"]).astype(np.float32)

        #Setup the host state object
        self.hostState = HostState(mass,pos,vel)

        #Load Body information
        for index,body in enumerate(json_data["bodies"]):
            self.hostState.add_body( Body(index, body) )

        #Setup and start the clock
        self.clock = pygame.time.Clock()
        self.time = 0


    def run(self):
        from gravutils.device import DeviceState
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

#------------------------------------------------------------------------
#--------------------------CAMERA CLASS----------------------------------
#------------------------------------------------------------------------
class Camera(object):
    """ The class for the camera """
    def __init__(self, loc, dim2, scale):

        #Position and direction of camera
        self.loc = loc
        self.dir = np.array([0,0,-1])

        self.dim2 = dim2
        self._sc = scale
        self._zm  = 1
        self.theta = 0;
        self.phi = 0;

    def translate(self, dp):
        self.loc += dp

    def scale(self, zoom):
        self._zm *= zoom

    def project(self, location):
        proj = np.dot(self.dir, location)/np.linalg.norm(self.dir)**2*self.dir

    def location_to_pixels(self, location):
        # return np.array([WIDTHD2,HEIGHTD2]) + zoom*(loc*DRAW_SCALE - np.array([WIDTHD2,HEIGHTD2]))
        return (location[:2]*self._sc - self.loc[:2])*self._zm + self.dim2 #Relative to center

    def draw_objects(self, hs, pygame, win):
        for body in hs.get_bodies():
            draw_pos = self.location_to_pixels(body.position(hs.pos))
            pygame.draw.circle(win, (255, 255, 255),
                (int(draw_pos[0]),
                int(draw_pos[1])),
                int(body.radius*self._zm), 0)

def main():
    global WIDTH, HEIGHT, HEIGHTD2, WIDTHD2, DT, DRAW_SCALE

    pygame.init()
    win=pygame.display.set_mode((WIDTH, HEIGHT))

    simulator = GravSim()
    simulator.start()

    camera = Camera(np.array([0, 0, 3]), np.array([WIDTHD2, HEIGHTD2]), DRAW_SCALE)

    keysPressed = defaultdict(bool)

    def ScanKeyboard():
        while True:
            # Update the keysPressed state:
            evt = pygame.event.poll()
            if evt.type == pygame.NOEVENT:
                break
            elif evt.type in [pygame.KEYDOWN, pygame.KEYUP]:
                keysPressed[evt.key] = evt.type == pygame.KEYDOWN

    # Zoom factor, changed at runtime via the '+' and '-' numeric keypad keys
    zoom = 1.0

    bClearScreen = True
    pygame.display.set_caption("Gravity simulation (SPACE: show orbits, up/down : zoom in/out)")

    states = []

    #start the clock
    render_clock = pygame.time.Clock()


    while True:
        pygame.display.flip()
        if bClearScreen:  # Show orbits or not?
            win.fill((0, 0, 0))
        win.lock()

        #Drawing of the bodies
        with simulator.phys_lock:
            camera.draw_objects(simulator.hostState, pygame, win)



        # Zoom in/out
        win.unlock()
        ScanKeyboard()

        # update zoom factor (numeric keypad +/- keys)
        if keysPressed[pygame.K_DOWN]:
            camera.scale(0.98)
        if keysPressed[pygame.K_UP]:
            camera.scale(1.02)
        if keysPressed[pygame.K_a]:
            camera.translate(np.array([-5, 0, 0]))
        if keysPressed[pygame.K_d]:
            camera.translate(np.array([5, 0, 0]))
        if keysPressed[pygame.K_w]:
            camera.translate(np.array([0, -5, 0]))
        if keysPressed[pygame.K_s]:
            camera.translate(np.array([0, 5, 0]))
        if keysPressed[pygame.K_ESCAPE]:
            simulator.stop()
            break
        if keysPressed[pygame.K_SPACE]:
            while keysPressed[pygame.K_SPACE]:
                ScanKeyboard()
            bClearScreen = not bClearScreen
            verb = "show" if bClearScreen else "hide"
            pygame.display.set_caption(
                "Gravity simulation (SPACE: {} orbits, up/down : zoom in/out)".format(verb))


        #Tick the clock
        render_clock.tick(50)

    print("Exit main thread")



if __name__ == "__main__":
    main()
