#!/usr/bin/env python

class Body(object):
    """
    Represent one body on the front-end
        Stores no positional data on it's own, instead stores a reference ID
    """
    def __init__(self, obj_id, json_data):
        self.obj_id = obj_id
        self.name = json_data["name"]
        self.radius = json_data["radius"]

    def __repr__(self):
        return 'ID:{ID} Name:{name} radius:{r}'.format(ID=self.obj_id, name=self.name, r=self.radius)

    def position(self, positions):
        return positions[self.obj_id]


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
        self.bodies = []

    def sync_with_device(self, dstate):
        self.mass = dstate.mass.get()
        self.pos = dstate.pos.get()
        self.vel = dstate.vel.get()

    def add_body(self, body):
        self.bodies.append(body)

    def get_bodies(self):
        return self.bodies
