import sys
import math
import time

#INTERFACE
import pygame
import pygame.gfxdraw
from collections import defaultdict
import threading

#BACKEND
import numpy as np

from gravutils.host import HostState, Body

#----WINDOW PARAMETERS
WIDTH, HEIGHT = 900, 600
WIDTHD2, HEIGHTD2 = WIDTH/2., HEIGHT/2.

import euclid
from euclid import Matrix4

#----DRAWING SCALE
DRAW_SCALE = 0.1

#----ANTIALIASING
AA = True

#------------------------------------------------------------------------
#--------------------------CAMERA CLASS----------------------------------
#------------------------------------------------------------------------
class Camera(object):
    """ The class for the camera """

    s_dir = np.array([0,0,-1])
    _zm  = 1
    theta = 0
    phi = 0

    rotm = np.zeros((4,4))

    def __init__(self, loc, dim2, scale):

        #Position and direction of camera
        self.loc = loc
        self.dir = self.s_dir

        self.dim2 = dim2
        self._sc = scale

        self.update_projection()

    def translate(self, dp):
        self.loc += dp
        self.update_projection()

    def scale(self, zoom):
        self._zm *= zoom

    def project(self, location):
        #proj = np.dot(self.dir, location)/np.linalg.norm(self.dir)**2*self.dir
        return np.dot(self.rotm, np.resize(location,(4,1)))

    def location_to_pixels(self, location):
        #print(self.project(location))
        # return np.array([WIDTHD2,HEIGHTD2]) + zoom*(loc*DRAW_SCALE - np.array([WIDTHD2,HEIGHTD2]))
        return (location[:2]*self._sc - self.loc[:2])*self._zm + self.dim2 #Relative to center

    def update_projection(self):
        # from http://www.euclideanspace.com/
        ch = math.cos(self.theta)
        sh = math.sin(self.theta)
        ca = math.cos(self.phi)
        sa = math.sin(self.phi)
        cb = 1
        sb = 0

        self.rotm[0,0] = ch * ca
        self.rotm[0,1] = sh * sb - ch * sa * cb
        self.rotm[0,2] = ch * sa * sb + sh * cb
        self.rotm[0,3] = self.loc[0]

        self.rotm[1,0] = sa
        self.rotm[1,1] = ca * cb
        self.rotm[1,2] = -ca * sb
        self.rotm[1,3] = self.loc[1]

        self.rotm[2,0] = -sh * ca
        self.rotm[2,1] = sh * sa * cb + ch * sb
        self.rotm[2,2] = -sh * sa * sb + ch * cb
        self.rotm[2,3] = self.loc[2]


    def draw_objects(self, win, hs, pygame):
        for body in hs.get_bodies():
            draw_pos = self.location_to_pixels(body.position(hs.pos))
            x = int(draw_pos[0])
            y = int(draw_pos[1])
            r = int(body.radius*self._zm)
            if(AA):
                pygame.gfxdraw.aacircle(win, x, y, r, (255, 255, 255))
                pygame.gfxdraw.filled_circle(win, x, y, r, (255, 255, 255))
            else:
                pygame.draw.circle(win, (255, 255, 255), (x, y), z, 0)


class SimRenderer(threading.Thread):
    """docstring for SimRenderer"""
    def __init__(self, simulator):
        super(SimRenderer, self).__init__()
        self.sim = simulator

        self.camera = Camera(np.array([0, 0, 3]), np.array([WIDTHD2, HEIGHTD2]), DRAW_SCALE)
        self.keysPressed = defaultdict(bool)

        # Zoom factor, changed at runtime via the '+' and '-' numeric keypad keys
        zoom = 1.0


    def scanKeyboard(self):
        while True:
            # Update the keysPressed state:
            evt = pygame.event.poll()
            if evt.type == pygame.NOEVENT:
                break
            elif evt.type in [pygame.KEYDOWN, pygame.KEYUP]:
                self.keysPressed[evt.key] = evt.type == pygame.KEYDOWN

    def run(self):
        pygame.init()
        win=pygame.display.set_mode((WIDTH, HEIGHT))

        bClearScreen = True
        pygame.display.set_caption("Gravity simulation (SPACE: show orbits, up/down : zoom in/out)")

        #start the clock
        render_clock = pygame.time.Clock()

        while True:
            pygame.display.flip()
            if bClearScreen:  # Show orbits or not?
                win.fill((0, 0, 0))
            win.lock()

            #Drawing of the bodies
            with self.sim.phys_lock:
                self.camera.draw_objects(win, self.sim.hostState, pygame)

            # Zoom in/out
            win.unlock()
            self.scanKeyboard()

            # update zoom factor (numeric keypad +/- keys)
            if self.keysPressed[pygame.K_DOWN]:
                self.camera.scale(0.98)
            if self.keysPressed[pygame.K_UP]:
                self.camera.scale(1.02)
            if self.keysPressed[pygame.K_a]:
                self.camera.translate(np.array([-5, 0, 0]))
            if self.keysPressed[pygame.K_d]:
                self.camera.translate(np.array([5, 0, 0]))
            if self.keysPressed[pygame.K_w]:
                self.camera.translate(np.array([0, -5, 0]))
            if self.keysPressed[pygame.K_s]:
                self.camera.translate(np.array([0, 5, 0]))
            if self.keysPressed[pygame.K_ESCAPE]:
                self.sim.stop()
                break
            if self.keysPressed[pygame.K_SPACE]:
                while self.keysPressed[pygame.K_SPACE]:
                    self.scanKeyboard()
                bClearScreen = not bClearScreen
                verb = "show" if bClearScreen else "hide"
                pygame.display.set_caption(
                    "Gravity simulation (SPACE: {} orbits, up/down : zoom in/out)".format(verb))

            #Tick the clock
            render_clock.tick(50)

        print("Exiting render thread")
