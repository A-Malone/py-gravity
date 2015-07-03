import pygame
import numpy as np
from collections import defaultdict
import sys
import math
import threading, time

#----WINDOW PARAMETERS
WIDTH, HEIGHT = 900, 600
WIDTHD2, HEIGHTD2 = WIDTH/2., HEIGHT/2.

#----DRAWING SCALE
DRAW_SCALE = 0.1


#Constant for Forest-Ruth
THETA = 1/(2 - 2**(1.0/3))

class State(object):
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

    def __init__(self, timestep=None):

        #Init the super class
        super(GravSim, self).__init__()

        #Create a lock object
        self.phys_lock = threading.Lock()
        self.stopped = False

        #Timstep
        self.dt = timestep or self.DT

        self.G = 6.67*10**(-11) * self.MASS_SACLE / self.LENGTH_SCALE **3 * (self.TIME_SCALE)**2 #(LU)^3/(MU)/(TU)^2

        #----Setup the bodies
        bodies = []
        sun = Body(0, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]),  332946, 25)
        bodies.append(sun)

        """
        earth = Body(100, np.array([1000.0,0.0]), np.array([0.0,math.sqrt(self.G*sun.mass/1000.0)]),  1, 2)
        vfunc = math.sqrt(self.G*sun.mass/1000.0)
        vreal = math.sqrt(-3.413706849457011E-03**2 + 1.685171267854315E-02**2 + -3.931485357309581E-07**2) * 1000 / 24.0 / 60.0
        arr = np.array([-5.910686063342294E+00,  2.917801312914639E+01, -6.807197200714606E-04])
        arr_2 = np.array([-3.413706849457011E-03,  1.685171267854315E-02, -3.931485357309581E-07])
        print(np.sqrt(arr.dot(arr))*10**3 / LENGTH_SCALE * TIME_SCALE)
        print(np.sqrt(arr.dot(arr))*10**3 / LENGTH_SCALE * TIME_SCALE)


        print(vfunc , vreal, vreal/vfunc)
        bodies.append(earth)
        """

        #SOLAR SYSTEM DATA
        #   ssd[i][2:0] => positions x,y,z in AU
        #   ssd[i][5:3] => velocity  x,y,z in AU/day
        ssd = [ (3.296437666226646E-01, -2.305331200568514E-01, -4.887672789792401E-02,  1.061082362225669E-02,  2.434704693325687E-02,  1.015757841274280E-03),
                (-7.154027730218678E-01,  3.337539622810829E-02,  4.177443581670932E-02, -1.086597212755203E-03, -2.028662925468785E-02, -2.152398820568715E-04),
                (9.861732313020960E-01,  1.809250582101355E-01, -1.391006750200533E-04, -3.413706849457011E-03,  1.685171267854315E-02, -3.931485357309581E-07),
                (5.484577138401596E-01, -1.305198171408829E+00, -4.084956238141536E-02,  1.344288466582773E-02,  6.612217615823506E-03, -1.915033347255382E-04),
                (-3.211787993621413E+00,  4.201506346143431E+00,  5.434137703482850E-02, -6.085188929003960E-03, -4.227184133398781E-03,  1.537379058870405E-04),
                (-5.788398185748002E+00, -8.069056761371796E+00,  3.706693349308769E-01,  4.228964967570854E-03, -3.266678393918157E-03, -1.117677552850386E-04),
                (1.939815666659398E+01,  4.928293091544985E+00, -2.330057456118213E-01, -9.972160175613773E-04,  3.628663423752756E-03,  2.631337365376518E-05),
                (2.741993616461276E+01, -1.210153890018418E+01, -3.827123750638945E-01,  1.246157252751333E-03,  2.890234553618175E-03, -8.815677984951996E-05),
                (7.122658572083311E+00, -3.192497674457871E+01,  1.355886722254575E+00,  3.132865137371930E-03,  5.617508561688919E-05, -9.039688277476739E-04)]

        ssm = [0.0558, 0.815 ,1.000,0.107,318,95.1,14.5,17.2,0.01]

        for index,i in enumerate(ssd):
            b = Body(index+1, np.array([i[0],i[1], i[2]])*1000.0, np.array([i[3],i[4], i[5]])* self.AU / self.LENGTH_SCALE / 24.0 / 60.0, ssm[index], 1)
            bodies.append(b)

        #Load data into the simulator
        pos = np.asarray(ssd)[:,:3]
        vel = np.asarray(ssd)[:,3:]
        mass = np.asarray(ssm)

        #self.bodies = bodies
        self.last_state = self.get_current_state()

        #Setup and start the clock
        self.clock = pygame.time.Clock()
        self.time = 0


    def run(self):
        print("Starting physics simulator")
        while(not self.stopped):
            with self.phys_lock:
                self.last_state = self.get_current_state()

            self.update(self.dt)
            self.time += self.dt
            #f(not self.time % (1440 * 60)):
            #    print(self.bodies[3].loc)
            self.clock.tick(50)
        print("Stopping physics simulator")

    def stop(self):
        self.stopped = True

    def get_current_state(self):
        return [x.current_state() for x in self.bodies]

    def update(self, dt):
        for body in self.bodies:
            self.update_body(body, self.last_state, dt)
            #print(body.loc)

    def update_body(self, body, states, dt):
        """ The physics engine itself """
        global THETA

        #Forest-Ruth - 4th order, symplectic
        body.loc = body.loc +          THETA  * dt/2 * body.vel
        body.vel = body.vel +          THETA  * dt   * self.acceleration(body.loc, body, states)
        body.loc = body.loc + (1   -   THETA) * dt/2 * body.vel
        body.vel = body.vel + (1   - 2*THETA) * dt   * self.acceleration(body.loc, body, states)
        body.loc = body.loc + (1   -   THETA) * dt/2 * body.vel
        body.vel = body.vel +          THETA  * dt   * self.acceleration(body.loc, body, states)
        body.loc = body.loc +          THETA  * dt/2 * body.vel

    def acceleration(self, pos, body, states):
        """ Provides acceleration that allows for the derivation of the slop"""
        acceleration = np.array([0.0,0.0,0.0])
        for obj in states:
            if obj.obj_id != body.obj_id:
                diff = obj.loc - pos
                acceleration += self.G * obj.mass / np.linalg.norm(diff)**3 * diff
        #print("{}, {}".format(body.vel, acceleration))
        return acceleration

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

    def draw_objects(self, states, pygame, win):
        for p in states:
            draw_pos = self.location_to_pixels(p.loc)
            pygame.draw.circle(win, (255, 255, 255),
                (int(draw_pos[0]),
                int(draw_pos[1])),
                int(p.radius*self._zm), 0)

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
            states = simulator.last_state

        camera.draw_objects(states, pygame, win)

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
