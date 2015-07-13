import sys
import math
import time

#INTERFACE
import pygame
from collections import defaultdict
import threading

#BACKEND
import numpy as np

from gravutils.host import HostState, Body

#----WINDOW PARAMETERS
WIDTH, HEIGHT = 900, 600
WIDTHD2, HEIGHTD2 = WIDTH/2., HEIGHT/2.

#----DRAWING SCALE
DRAW_SCALE = 0.1

#----ANTIALIASING
AA = False

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
            x = (int(draw_pos[0])
            y = (int(draw_pos[1])
            z = int(body.radius*self._zm)
            if(AA):
                pygame.gfxdraw.aacircle(surf, x, y, 30, (255, 0, 0))
                pygame.gfxdraw.filled_circle(surf, x, y, 30, (255, 0, 0))
            else:
                pygame.draw.circle(win, (255, 255, 255), x,y, z, 0)


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
                self.camera.draw_objects(self.sim.hostState, pygame, win)

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
