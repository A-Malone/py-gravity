class State(object):
    """Used to represent the physical state of the system"""
    mass = None
    pos = None
    vel = None

    def __init__(self, mass, pos, vel):
        super(State, self).__init__()
